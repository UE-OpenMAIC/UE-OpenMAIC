/**
 * Media Generation Orchestrator
 *
 * Dispatches media generation API calls for all mediaGenerations across outlines.
 * Runs entirely on the frontend — calls /api/generate/image and /api/generate/video,
 * fetches result blobs, stores in IndexedDB, and updates the Zustand store.
 */

import { useMediaGenerationStore } from '@/lib/store/media-generation';
import { useSettingsStore } from '@/lib/store/settings';
import { db, mediaFileKey } from '@/lib/utils/database';
import type { SceneOutline } from '@/lib/types/generation';
import type { MediaGenerationRequest } from '@/lib/media/types';
import { createLogger } from '@/lib/logger';

const log = createLogger('MediaOrchestrator');
const activeMediaRequests = new Set<string>();
const MAX_AI_IMAGES_PER_COURSE = 10;
const MAX_AI_VIDEOS_PER_COURSE = 2;

/** Error with a structured errorCode from the API */
class MediaApiError extends Error {
  errorCode?: string;
  constructor(message: string, errorCode?: string) {
    super(message);
    this.errorCode = errorCode;
  }
}
/**
 * Launch media generation for all mediaGenerations declared in outlines.
 * Runs in parallel with content/action generation — does not block.
 */
export async function generateMediaForOutlines(
  outlines: SceneOutline[],
  stageId: string,
  abortSignal?: AbortSignal,
): Promise<void> {
  const settings = useSettingsStore.getState();
  const store = useMediaGenerationStore.getState();

  // Collect all media requests
  const candidates: MediaGenerationRequest[] = [];
  const seenElementIds = new Set<string>();
  const skippedBySettings: string[] = [];
  const skippedFailed: string[] = [];
  const skippedByBudget: string[] = [];
  let imageBudgetUsed = 0;
  let videoBudgetUsed = 0;
  for (const outline of outlines) {
    if (!outline.mediaGenerations) continue;
    for (const mg of outline.mediaGenerations) {
      if (seenElementIds.has(mg.elementId)) continue;
      seenElementIds.add(mg.elementId);

      // Filter by enabled flags
      if (mg.type === 'image' && !settings.imageGenerationEnabled) {
        skippedBySettings.push(mg.elementId);
        continue;
      }
      if (mg.type === 'video' && !settings.videoGenerationEnabled) {
        skippedBySettings.push(mg.elementId);
        continue;
      }
      if (mg.type === 'image') {
        if (imageBudgetUsed >= MAX_AI_IMAGES_PER_COURSE) {
          skippedByBudget.push(mg.elementId);
          continue;
        }
        imageBudgetUsed += 1;
      }
      if (mg.type === 'video') {
        if (videoBudgetUsed >= MAX_AI_VIDEOS_PER_COURSE) {
          skippedByBudget.push(mg.elementId);
          continue;
        }
        videoBudgetUsed += 1;
      }
      const existing = store.getTask(mg.elementId);
      if (
        existing?.status === 'done' ||
        existing?.status === 'pending' ||
        existing?.status === 'generating' ||
        activeMediaRequests.has(mg.elementId)
      ) {
        continue;
      }
      if (existing?.status === 'failed') {
        skippedFailed.push(mg.elementId);
        continue;
      }
      candidates.push(mg);
    }
  }

  if (skippedBySettings.length > 0) {
    log.info(
      `Skipped ${skippedBySettings.length} media request(s) because generation is disabled: ${skippedBySettings.join(', ')}`,
    );
  }

  if (skippedByBudget.length > 0) {
    log.info(
      `Skipped ${skippedByBudget.length} media request(s) beyond course budget (${MAX_AI_IMAGES_PER_COURSE} images, ${MAX_AI_VIDEOS_PER_COURSE} video): ${skippedByBudget.join(', ')}`,
    );
  }

  const allRequests: MediaGenerationRequest[] = [];
  const skippedPersistedDone: string[] = [];
  const skippedPersistedFailed: string[] = [];

  if (candidates.length > 0) {
    const persistedRecords = await db.mediaFiles.bulkGet(
      candidates.map((req) => mediaFileKey(stageId, req.elementId)),
    );

    for (let i = 0; i < candidates.length; i += 1) {
      const req = candidates[i];
      const persisted = persistedRecords[i];

      // A persisted record means this placeholder has already been handled for this course.
      // Do not spend another image/video API call unless the user explicitly clicks retry.
      if (persisted) {
        if (persisted.error) {
          skippedPersistedFailed.push(req.elementId);
        } else {
          skippedPersistedDone.push(req.elementId);
        }
        continue;
      }

      allRequests.push(req);
    }
  }

  if (skippedPersistedDone.length > 0) {
    log.info(
      `Skipped ${skippedPersistedDone.length} persisted media result(s): ${skippedPersistedDone.join(', ')}`,
    );
  }

  if (skippedFailed.length > 0 || skippedPersistedFailed.length > 0) {
    const failedIds = [...skippedFailed, ...skippedPersistedFailed];
    log.info(
      `Skipped ${failedIds.length} failed media request(s). Use explicit retry to regenerate: ${failedIds.join(', ')}`,
    );
  }

  if (allRequests.length === 0) {
    log.info('No pending media requests to generate.');
    return;
  }

  log.info(
    `Queueing ${allRequests.length} media request(s): ${allRequests
      .map((req) => `${req.type}:${req.elementId}`)
      .join(', ')}`,
  );

  // Enqueue all as pending
  useMediaGenerationStore.getState().enqueueTasks(stageId, allRequests);

  // Process requests serially — image/video APIs have limited concurrency
  for (const req of allRequests) {
    if (abortSignal?.aborted) break;
    await generateSingleMedia(req, stageId, abortSignal);
  }
}

/**
 * Retry a single failed media task.
 */
export async function retryMediaTask(elementId: string): Promise<void> {
  const store = useMediaGenerationStore.getState();
  const task = store.getTask(elementId);
  if (!task || task.status !== 'failed') return;

  // Check if the corresponding generation type is still enabled in global settings
  const settings = useSettingsStore.getState();
  if (task.type === 'image' && !settings.imageGenerationEnabled) {
    store.markFailed(elementId, 'Generation disabled', 'GENERATION_DISABLED');
    return;
  }
  if (task.type === 'video' && !settings.videoGenerationEnabled) {
    store.markFailed(elementId, 'Generation disabled', 'GENERATION_DISABLED');
    return;
  }

  // Remove persisted failure record from DB so a fresh result can be written
  const dbKey = mediaFileKey(task.stageId, elementId);
  await db.mediaFiles.delete(dbKey).catch(() => {});

  store.markPendingForRetry(elementId);
  await generateSingleMedia(
    {
      type: task.type,
      prompt: task.prompt,
      elementId: task.elementId,
      aspectRatio: task.params.aspectRatio as MediaGenerationRequest['aspectRatio'],
      style: task.params.style,
      duration: task.params.duration,
      resolution: task.params.resolution as MediaGenerationRequest['resolution'],
      generateAudio: task.params.generateAudio,
    },
    task.stageId,
  );
}

// ==================== Internal ====================

async function generateSingleMedia(
  req: MediaGenerationRequest,
  stageId: string,
  abortSignal?: AbortSignal,
): Promise<void> {
  if (activeMediaRequests.has(req.elementId)) return;
  activeMediaRequests.add(req.elementId);
  const store = useMediaGenerationStore.getState();
  store.markGenerating(req.elementId);

  try {
    let resultUrl: string;
    let posterUrl: string | undefined;
    let mimeType: string;

    if (req.type === 'image') {
      const result = await callImageApi(req, abortSignal);
      resultUrl = result.url;
      mimeType = 'image/png';
    } else {
      const result = await callVideoApi(req, abortSignal);
      resultUrl = result.url;
      posterUrl = result.poster;
      mimeType = 'video/mp4';
    }

    if (abortSignal?.aborted) return;

    // Fetch blob from URL
    const blob = await fetchAsBlob(resultUrl);
    const posterBlob = posterUrl ? await fetchAsBlob(posterUrl).catch(() => undefined) : undefined;

    // Store in IndexedDB
    await db.mediaFiles.put({
      id: mediaFileKey(stageId, req.elementId),
      stageId,
      type: req.type,
      blob,
      mimeType,
      size: blob.size,
      poster: posterBlob,
      prompt: req.prompt,
      params: JSON.stringify({
        aspectRatio: req.aspectRatio,
        style: req.style,
        duration: req.duration,
        resolution: req.resolution,
        generateAudio: req.generateAudio,
      }),
      createdAt: Date.now(),
    });

    // Update store with object URL
    const objectUrl = URL.createObjectURL(blob);
    const posterObjectUrl = posterBlob ? URL.createObjectURL(posterBlob) : undefined;
    useMediaGenerationStore.getState().markDone(req.elementId, objectUrl, posterObjectUrl);
  } catch (err) {
    if (abortSignal?.aborted) return;
    const message = err instanceof Error ? err.message : String(err);
    const errorCode = err instanceof MediaApiError ? err.errorCode : undefined;
    log.error(`Failed ${req.elementId}:`, message);
    useMediaGenerationStore.getState().markFailed(req.elementId, message, errorCode);

    // Persist non-retryable failures to IndexedDB so they survive page refresh
    if (errorCode) {
      await db.mediaFiles
        .put({
          id: mediaFileKey(stageId, req.elementId),
          stageId,
          type: req.type,
          blob: new Blob(), // empty placeholder
          mimeType: req.type === 'image' ? 'image/png' : 'video/mp4',
          size: 0,
          prompt: req.prompt,
          params: JSON.stringify({
            aspectRatio: req.aspectRatio,
            style: req.style,
            duration: req.duration,
            resolution: req.resolution,
            generateAudio: req.generateAudio,
          }),
          error: message,
          errorCode,
          createdAt: Date.now(),
        })
        .catch(() => {}); // best-effort
    }
  } finally {
    activeMediaRequests.delete(req.elementId);
  }
}

async function callImageApi(
  req: MediaGenerationRequest,
  abortSignal?: AbortSignal,
): Promise<{ url: string }> {
  const settings = useSettingsStore.getState();
  const providerConfig = settings.imageProvidersConfig?.[settings.imageProviderId];

  const response = await fetch('/api/generate/image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-image-provider': settings.imageProviderId || '',
      'x-image-model': settings.imageModelId || '',
      'x-api-key': providerConfig?.apiKey || '',
      'x-base-url': providerConfig?.baseUrl || '',
    },
    body: JSON.stringify({
      prompt: req.prompt,
      aspectRatio: req.aspectRatio,
      style: req.style,
    }),
    signal: abortSignal,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new MediaApiError(data.error || `Image API returned ${response.status}`, data.errorCode);
  }

  const data = await response.json();
  if (!data.success)
    throw new MediaApiError(data.error || 'Image generation failed', data.errorCode);

  // Result may have url or base64
  const url =
    data.result?.url || (data.result?.base64 ? `data:image/png;base64,${data.result.base64}` : '');
  if (!url) throw new Error('No image URL in response');
  return { url };
}

async function callVideoApi(
  req: MediaGenerationRequest,
  abortSignal?: AbortSignal,
): Promise<{ url: string; poster?: string }> {
  const settings = useSettingsStore.getState();
  const providerConfig = settings.videoProvidersConfig?.[settings.videoProviderId];
  const seedreamConfig = settings.imageProvidersConfig?.seedream;
  const apiKey =
    providerConfig?.apiKey ||
    (settings.videoProviderId === 'seedance' ? seedreamConfig?.apiKey : '') ||
    '';
  const baseUrl =
    providerConfig?.baseUrl ||
    (settings.videoProviderId === 'seedance' ? seedreamConfig?.baseUrl : '') ||
    '';

  const response = await fetch('/api/generate/video', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-video-provider': settings.videoProviderId || '',
      'x-video-model': settings.videoModelId || '',
      'x-api-key': apiKey,
      'x-base-url': baseUrl,
    },
    body: JSON.stringify({
      prompt: req.prompt,
      aspectRatio: req.aspectRatio,
      duration: req.duration,
      resolution: req.resolution,
      generateAudio: req.generateAudio ?? true,
    }),
    signal: abortSignal,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new MediaApiError(data.error || `Video API returned ${response.status}`, data.errorCode);
  }

  const data = await response.json();
  if (!data.success)
    throw new MediaApiError(data.error || 'Video generation failed', data.errorCode);

  const url = data.result?.url;
  if (!url) throw new Error('No video URL in response');
  return { url, poster: data.result?.poster };
}

async function fetchAsBlob(url: string): Promise<Blob> {
  // For data URLs, convert directly
  if (url.startsWith('data:')) {
    const res = await fetch(url);
    return res.blob();
  }
  // For remote URLs, proxy through our server to bypass CORS restrictions
  if (url.startsWith('http://') || url.startsWith('https://')) {
    const res = await fetch('/api/proxy-media', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Proxy fetch failed: ${res.status}`);
    }
    return res.blob();
  }
  // Relative URLs (shouldn't happen, but handle gracefully)
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch blob: ${res.status}`);
  return res.blob();
}
