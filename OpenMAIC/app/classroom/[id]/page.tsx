'use client';

import { Stage } from '@/components/stage';
import { ThemeProvider } from '@/lib/hooks/use-theme';
import { useStageStore } from '@/lib/store';
import { loadImageMapping } from '@/lib/utils/image-storage';
import { useEffect, useRef, useState, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { useSceneGenerator } from '@/lib/hooks/use-scene-generator';
import { useMediaGenerationStore } from '@/lib/store/media-generation';
import { useWhiteboardHistoryStore } from '@/lib/store/whiteboard-history';
import { createLogger } from '@/lib/logger';
import { MediaStageProvider } from '@/lib/contexts/media-stage-context';
import { generateMediaForOutlines } from '@/lib/media/media-orchestrator';
// 导入教师动作
// import { TeacherPanel, type TeacherStatus } from './components/TeacherPanel';
// 导入UE教师
import type { TeacherStatus } from './components/TeacherPanel';
import { UEViewport } from './components/UEViewport';
import type { Scene } from '@/lib/types/stage';
import type { Action } from '@/lib/types/action';

const log = createLogger('Classroom');

type ClassroomAction = {
  type?: string;
  mocapId?: string;
  mocapAction?: string;
  mocapCoarse?: string;
  confidenceScore?: number;
  sourceText?: string;
  teacherStatus?: TeacherStatus;
  ueAction?: string;
  writeElementId?: string;
};

function getMocapTeacherStatus(action?: ClassroomAction | null): TeacherStatus | null {
  if (!action?.mocapId) return null;
  return normalizeTeacherStatus(action.ueAction || action.teacherStatus || action.mocapId);
}

function normalizeTeacherStatus(value?: string | null): TeacherStatus {
  switch (value) {
    case 'Idle':
    case 'idle':
      return 'Idle';
    case 'Talking':
    case 'talking':
      return 'Talking';
    case 'Pointing':
    case 'pointing':
      return 'Pointing';
    case 'Thinking':
    case 'thinking':
      return 'Thinking';
    case 'MC05':
    case 'MC06':
    case 'MC07':
    case 'MC08':
    case 'MC09':
    case 'MC10':
    case 'MC11':
    case 'MC12':
    case 'MC13':
    case 'MC14':
    case 'MC15':
      return value;
    default:
      return 'Talking';
  }
}

function restoreBoardWritingText(scenes: Scene[]): Scene[] {
  let changed = false;

  const restoredScenes = scenes.map((scene) => {
    if (scene.type !== 'slide' || scene.content.type !== 'slide') return scene;

    const normalizedActions = normalizeBoardWritingActions(scene.actions || []);
    if (normalizedActions !== scene.actions) {
      changed = true;
    }

    const writeContentByElementId = new Map<string, string>();
    for (const action of normalizedActions || []) {
      if (
        action.type === 'speech' &&
        action.writeElementId &&
        action.writeContent &&
        action.writeContent.trim()
      ) {
        writeContentByElementId.set(action.writeElementId, action.writeContent);
      }
    }
    if (writeContentByElementId.size === 0) return scene;

    const elements = scene.content.canvas.elements.map((element) => {
      if (element.type !== 'text') return element;
      const restoredContent = writeContentByElementId.get(element.id);
      if (!restoredContent) return element;
      if (element.content && element.content.trim()) return element;

      changed = true;
      return {
        ...element,
        content: restoredContent,
        opacity: 1,
      };
    });

    return {
      ...scene,
      actions: normalizedActions,
      content: {
        ...scene.content,
        canvas: {
          ...scene.content.canvas,
          elements,
        },
      },
    };
  });

  return changed ? restoredScenes : scenes;
}

function normalizeBoardWritingActions(actions: Action[]): Action[] {
  let changed = false;
  const normalized: Action[] = [];

  for (const action of actions) {
    if (action.type !== 'speech' || !action.writeElementId || !action.writeContent) {
      normalized.push(action);
      continue;
    }

    const fullText = stripHtmlText(action.writeContent);
    const looksLikeOldFullBoxAction = fullText && fullText === action.text;
    if (!looksLikeOldFullBoxAction) {
      normalized.push(action);
      continue;
    }

    const lines = extractTextLines(action.writeContent);
    if (lines.length <= 1) {
      normalized.push(action);
      continue;
    }

    changed = true;
    const fontSize = action.writeFontSize ?? extractFontSize(action.writeContent) ?? 22;
    lines.forEach((line, index) => {
      normalized.push({
        ...action,
        id: `${action.id}_line_${index + 1}`,
        text: line,
        mocapSelectorText: `板书书写：${line}`,
        writeContent: makeLinesHtml(lines.slice(0, index + 1), fontSize),
      });
    });
  }

  return changed ? normalized : actions;
}

function extractTextLines(html: string): string[] {
  const paragraphMatches = [...html.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/gi)];
  const lines =
    paragraphMatches.length > 0
      ? paragraphMatches.map((match) => stripHtmlText(match[1]))
      : stripHtmlText(html).split(/[。；;.\n]/);

  return lines.map((line) => line.trim()).filter(Boolean);
}

function makeLinesHtml(lines: string[], fontSize: number): string {
  return lines
    .map((line) => `<p style="font-size:${fontSize}px;">${escapeHtml(line)}</p>`)
    .join('');
}

function stripHtmlText(html: string): string {
  return html
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<[^>]*>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/\s+/g, ' ')
    .trim();
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function extractFontSize(html: string): number | null {
  const match = html.match(/font-size\s*:\s*(\d+(?:\.\d+)?)px/i);
  return match ? Number(match[1]) : null;
}

export default function ClassroomDetailPage() {
  // 动作识别ycf
  const { scenes, currentSceneId } = useStageStore();
  const currentScene = scenes.find((scene) => scene.id === currentSceneId) || null;

  const params = useParams();
  const classroomId = params?.id as string;

  const { loadFromStorage } = useStageStore();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [teacherStatus, setTeacherStatus] = useState<TeacherStatus>('Idle');

  const ueStreamUrl = 'http://127.0.0.1/player.html';
  const ueConnected = true;

  // 新增：拿到 UE iframe
  const ueIframeRef = useRef<HTMLIFrameElement | null>(null);

  //存储当前正在执行的动作
  const [currentAction, setCurrentAction] = useState<ClassroomAction | null>(null);
  const [currentMocapAction, setCurrentMocapAction] = useState<ClassroomAction | null>(null);

  const generationStartedRef = useRef(false);

  const { generateRemaining, retrySingleOutline, stop } = useSceneGenerator({
    onComplete: () => {
      log.info('[Classroom] All scenes generated');
    },
  });

  const loadClassroom = useCallback(async () => {
    try {
      await loadFromStorage(classroomId);

      // If IndexedDB had no data, try server-side storage (API-generated classrooms)
      if (!useStageStore.getState().stage) {
        log.info('No IndexedDB data, trying server-side storage for:', classroomId);
        try {
          const res = await fetch(`/api/classroom?id=${encodeURIComponent(classroomId)}`);
          if (res.ok) {
            const json = await res.json();
            if (json.success && json.classroom) {
              const { stage, scenes } = json.classroom;
              useStageStore.getState().setStage(stage);
              useStageStore.setState({
                scenes,
                currentSceneId: scenes[0]?.id ?? null,
              });
              log.info('Loaded from server-side storage:', classroomId);
            }
          }
        } catch (fetchErr) {
          log.warn('Server-side storage fetch failed:', fetchErr);
        }
      }

      // Restore completed media generation tasks from IndexedDB
      const restoredScenes = restoreBoardWritingText(useStageStore.getState().scenes);
      if (restoredScenes !== useStageStore.getState().scenes) {
        useStageStore.getState().setScenes(restoredScenes);
      }

      // Restore completed media generation tasks from IndexedDB
      await useMediaGenerationStore.getState().restoreFromDB(classroomId);
      // Restore agents for this stage
      const { loadGeneratedAgentsForStage, useAgentRegistry } =
        await import('@/lib/orchestration/registry/store');
      const generatedAgentIds = await loadGeneratedAgentsForStage(classroomId);
      const { useSettingsStore } = await import('@/lib/store/settings');
      if (generatedAgentIds.length > 0) {
        // Auto mode — use generated agents from IndexedDB
        useSettingsStore.getState().setAgentMode('auto');
        useSettingsStore.getState().setSelectedAgentIds(generatedAgentIds);
      } else {
        // Preset mode — restore agent IDs saved in the stage at creation time.
        // Filter out any stale generated IDs that may have been persisted before
        // the bleed-fix, so they don't resolve against a leftover registry entry.
        const stage = useStageStore.getState().stage;
        const stageAgentIds = stage?.agentIds;
        const registry = useAgentRegistry.getState();
        const cleanIds = stageAgentIds?.filter((id) => {
          const a = registry.getAgent(id);
          return a && !a.isGenerated;
        });
        useSettingsStore.getState().setAgentMode('preset');
        useSettingsStore
          .getState()
          .setSelectedAgentIds(
            cleanIds && cleanIds.length > 0 ? cleanIds : ['default-1', 'default-2', 'default-3'],
          );
      }
    } catch (error) {
      log.error('Failed to load classroom:', error);
      setError(error instanceof Error ? error.message : 'Failed to load classroom');
    } finally {
      setLoading(false);
    }
  }, [classroomId, loadFromStorage]);

  useEffect(() => {
    // Reset loading state on course switch to unmount Stage during transition,
    // preventing stale data from syncing back to the new course
    setLoading(true);
    setError(null);
    generationStartedRef.current = false;

    // Clear previous classroom's media tasks to prevent cross-classroom contamination.
    // Placeholder IDs (gen_img_1, gen_vid_1) are NOT globally unique across stages,
    // so stale tasks from a previous classroom would shadow the new one's.
    const mediaStore = useMediaGenerationStore.getState();
    mediaStore.revokeObjectUrls();
    useMediaGenerationStore.setState({ tasks: {} });

    // Clear whiteboard history to prevent snapshots from a previous course leaking in.
    useWhiteboardHistoryStore.getState().clearHistory();

    loadClassroom();

    // Cancel ongoing generation when classroomId changes or component unmounts
    return () => {
      stop();
    };
  }, [classroomId, loadClassroom, stop]);

  // Auto-resume generation for pending outlines
  useEffect(() => {
    if (loading || error || generationStartedRef.current) return;

    const state = useStageStore.getState();
    const { outlines, scenes, stage } = state;

    // Check if there are pending outlines
    const completedOrders = new Set(scenes.map((s) => s.order));
    const hasPending = outlines.some((o) => !completedOrders.has(o.order));

    if (hasPending && stage) {
      generationStartedRef.current = true;

      // Load generation params from sessionStorage (stored by generation-preview before navigating)
      const genParamsStr = sessionStorage.getItem('generationParams');
      const params = genParamsStr ? JSON.parse(genParamsStr) : {};

      // Reconstruct imageMapping from IndexedDB using pdfImages storageIds
      const storageIds = (params.pdfImages || [])
        .map((img: { storageId?: string }) => img.storageId)
        .filter(Boolean);

      // 读取老师状态
      loadImageMapping(storageIds).then((imageMapping) => {
        generateRemaining({
          pdfImages: params.pdfImages,
          imageMapping,
          stageInfo: {
            name: stage.name || '',
            description: stage.description,
            language: stage.language,
            style: stage.style,
          },
          agents: params.agents,
          userProfile: params.userProfile,
        });
      });
    } else if (outlines.length > 0 && stage) {
      // All scenes are generated, but some media may not have finished.
      // Resume media generation for any tasks not yet in IndexedDB.
      // generateMediaForOutlines skips already-completed tasks automatically.
      generationStartedRef.current = true;
      generateMediaForOutlines(outlines, stage.id).catch((err) => {
        log.warn('[Classroom] Media generation resume error:', err);
      });
    }
  }, [loading, error, generateRemaining]);
  useEffect(() => {
    const mocapStatus = getMocapTeacherStatus(currentMocapAction);

    console.log('[Classroom] status effect input', {
      currentAction,
      currentMocapAction,
      currentSceneType: currentScene?.type ?? null,
      currentSceneId: currentScene?.id ?? null,
      prevTeacherStatus: teacherStatus,
    });

    if (currentAction?.type === 'mocap') {
      const status = getMocapTeacherStatus(currentAction) || 'Talking';
      console.log('[Classroom] -> setTeacherStatus(from mocap)', {
        mocapId: currentAction.mocapId,
        mocapAction: currentAction.mocapAction,
        status,
      });
      setTeacherStatus(status);
      return;
    }

    if (mocapStatus) {
      console.log('[Classroom] -> keep mocap teacherStatus', {
        mocapId: currentMocapAction?.mocapId,
        mocapAction: currentMocapAction?.mocapAction,
        status: mocapStatus,
      });
      setTeacherStatus(mocapStatus);
      return;
    }

    if (currentAction?.type === 'speech' && currentAction.writeElementId) {
      console.log('[Classroom] -> keep teacherStatus, reason=board writing speech');
      return;
    }

    if (currentAction?.type === 'speech') {
      console.log('[Classroom] -> setTeacherStatus(Talking)');
      setTeacherStatus('Talking');
      return;
    }

    if (
      currentAction?.type === 'highlight' ||
      currentAction?.type === 'focus' ||
      currentAction?.type === 'point'
    ) {
      console.log('[Classroom] -> setTeacherStatus(Pointing)');
      setTeacherStatus('Pointing');
      return;
    }

    if (currentAction?.type === 'pause' || currentAction?.type === 'wait') {
      console.log('[Classroom] -> setTeacherStatus(Thinking)');
      setTeacherStatus('Thinking');
      return;
    }

    if (!currentScene) {
      console.log('[Classroom] -> setTeacherStatus(Idle), reason=no currentScene');
      setTeacherStatus('Idle');
      return;
    }

    if (currentScene.type === 'slide') {
      console.log('[Classroom] -> setTeacherStatus(Idle), reason=slide fallback');
      setTeacherStatus('Idle');
      return;
    }

    if (currentScene.type === 'quiz') {
      console.log('[Classroom] -> setTeacherStatus(Thinking), reason=quiz fallback');
      setTeacherStatus('Thinking');
      return;
    }

    if (currentScene.type === 'interactive' || currentScene.type === 'pbl') {
      console.log('[Classroom] -> setTeacherStatus(Pointing), reason=interactive/pbl fallback');
      setTeacherStatus('Pointing');
      return;
    }

    console.log('[Classroom] -> setTeacherStatus(Idle), reason=default fallback');
    setTeacherStatus('Idle');
  }, [currentAction, currentMocapAction, currentScene]);

  useEffect(() => {
    console.log('[Classroom] teacherStatus changed ->', teacherStatus);
  }, [teacherStatus]);

  useEffect(() => {
    const iframeWindow = ueIframeRef.current?.contentWindow;
    if (!iframeWindow) return;

    const mocapStatus = getMocapTeacherStatus(currentMocapAction);

    const payload = {
      source: 'openmaic',
      type: 'teacher_status',
      status: mocapStatus || normalizeTeacherStatus(teacherStatus),
      ...(currentMocapAction?.mocapId
        ? {
            mocapId: currentMocapAction.mocapId,
            mocapAction: currentMocapAction.mocapAction,
            ueAction: mocapStatus,
            reason: 'mocap-status-priority',
          }
        : {}),
      ts: Date.now(),
    };

    console.log('[Classroom] postMessage -> UE iframe', payload);

    iframeWindow.postMessage(payload, 'http://127.0.0.1');
  }, [teacherStatus, currentMocapAction]);

  useEffect(() => {
    if (currentAction?.type !== 'mocap') return;

    const iframeWindow = ueIframeRef.current?.contentWindow;
    if (!iframeWindow) return;

    const payload = {
      source: 'openmaic',
      type: 'teacher_mocap',
      status: getMocapTeacherStatus(currentAction) || 'Talking',
      mocapId: currentAction.mocapId,
      mocapAction: currentAction.mocapAction,
      mocapCoarse: currentAction.mocapCoarse,
      confidenceScore: currentAction.confidenceScore,
      sourceText: currentAction.sourceText,
      teacherStatus: getMocapTeacherStatus(currentAction) || 'Talking',
      ueAction: getMocapTeacherStatus(currentAction) || 'Talking',
      ts: Date.now(),
    };

    console.log('[Classroom] postMessage mocap -> UE iframe', payload);

    iframeWindow.postMessage(payload, 'http://127.0.0.1');
  }, [currentAction]);

  const handleActionChange = useCallback((action: ClassroomAction | null) => {
    setCurrentAction(action);
    if (action?.type === 'mocap') {
      setCurrentMocapAction(action);
    }
  }, []);

  return (
    <ThemeProvider>
      <MediaStageProvider value={classroomId}>
        <div className="h-screen flex flex-col overflow-hidden">
          {loading ? (
            <div className="flex-1 flex items-center justify-center bg-gray-50 dark:bg-gray-900">
              <div className="text-center text-muted-foreground">
                <p>Loading classroom...</p>
              </div>
            </div>
          ) : error ? (
            <div className="flex-1 flex items-center justify-center bg-gray-50 dark:bg-gray-900">
              <div className="text-center">
                <p className="text-destructive mb-4">Error: {error}</p>
                <button
                  onClick={() => {
                    setError(null);
                    setLoading(true);
                    loadClassroom();
                  }}
                  className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : (
            <Stage
              onRetryOutline={retrySingleOutline}
              onActionChange={handleActionChange}
              leftPanel={
                <UEViewport
                  teacherStatus={teacherStatus}
                  streamUrl={ueStreamUrl}
                  connected={ueConnected}
                  iframeRef={ueIframeRef}
                  mocapInfo={{
                    mocapId: currentMocapAction?.mocapId,
                    mocapAction: currentMocapAction?.mocapAction,
                    mappedAction: normalizeTeacherStatus(
                      currentMocapAction?.teacherStatus || teacherStatus,
                    ),
                    ueAction: normalizeTeacherStatus(
                      currentMocapAction?.ueAction ||
                        currentMocapAction?.teacherStatus ||
                        teacherStatus,
                    ),
                  }}
                />
              }
            />
          )}
        </div>
      </MediaStageProvider>
    </ThemeProvider>
  );
}
