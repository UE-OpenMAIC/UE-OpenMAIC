/**
 * Scene Outlines Streaming API (SSE)
 *
 * Streams outline generation via Server-Sent Events.
 * Emits individual outline objects as they're parsed from the LLM response,
 * so the frontend can display them incrementally.
 *
 * SSE events:
 *   { type: 'outline', data: SceneOutline, index: number }
 *   { type: 'done', outlines: SceneOutline[] }
 *   { type: 'error', error: string }
 */

import { NextRequest } from 'next/server';
import { streamLLM } from '@/lib/ai/llm';
import { buildPrompt, PROMPT_IDS } from '@/lib/generation/prompts';
import {
  formatImageDescription,
  formatImagePlaceholder,
  buildVisionUserContent,
  uniquifyMediaElementIds,
  formatTeacherPersonaForPrompt,
} from '@/lib/generation/generation-pipeline';
import type { AgentInfo } from '@/lib/generation/generation-pipeline';
import { MAX_PDF_CONTENT_CHARS, MAX_VISION_IMAGES } from '@/lib/constants/generation';
import { nanoid } from 'nanoid';
import type {
  UserRequirements,
  PdfImage,
  SceneOutline,
  ImageMapping,
} from '@/lib/types/generation';
import { apiError } from '@/lib/server/api-response';
import { createLogger } from '@/lib/logger';
import { resolveModelFromHeaders } from '@/lib/server/resolve-model';
import { ensureContentGroundedMediaPrompts } from '@/lib/generation/media-prompt-policy';
const log = createLogger('Outlines Stream');
const MAX_AI_IMAGES_PER_COURSE = 10;
const MAX_BLACKBOARD_LINES = 4;

const QUIZ_ASSESSMENT_TERMS = [
  '小测',
  '测验',
  '测试',
  '检测',
  '习题',
  '练习题',
  '选择题',
  '判断题',
  '填空题',
  '知识检查',
  'knowledge check',
  'quiz',
  'test',
  'assessment',
  'a/b/c/d',
];

function containsAssessmentTerm(text: string): boolean {
  const lower = text.toLowerCase();
  return QUIZ_ASSESSMENT_TERMS.some((term) => lower.includes(term));
}
function stripAssessmentTerms(text: string): string {
  let next = text;
  QUIZ_ASSESSMENT_TERMS.forEach((term) => {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    next = next.replace(new RegExp(escaped, 'gi'), '');
  });
  return next
    .replace(/\s+/g, ' ')
    .replace(/[：:，,、。；;]+$/g, '')
    .trim();
}

function trimBlackboardLine(line: string): string {
  const compact = line.replace(/\s+/g, ' ').trim();
  const hasCjk = /[\u3400-\u9fff]/.test(compact);
  const maxLength = hasCjk ? 18 : 58;
  return compact.length > maxLength ? `${compact.slice(0, maxLength - 1)}…` : compact;
}

function normalizeQuizOutlineAsBlackboard(outline: SceneOutline): SceneOutline {
  if (outline.type !== 'quiz') return outline;

  const baseTopic =
    stripAssessmentTerms(outline.title) ||
    stripAssessmentTerms(outline.description) ||
    stripAssessmentTerms((outline.keyPoints || []).join('，')) ||
    '核心概念';

  const cleanKeyPoints = (outline.keyPoints || [])
    .map((line) => stripAssessmentTerms(line))
    .filter((line) => line.length > 0 && !containsAssessmentTerm(line));

  const keyPoints =
    cleanKeyPoints.length > 0
      ? cleanKeyPoints.map(trimBlackboardLine).slice(0, MAX_BLACKBOARD_LINES)
      : [trimBlackboardLine(`${baseTopic} 的定义`), trimBlackboardLine(`${baseTopic} 的判定条件`), trimBlackboardLine(`${baseTopic} 的典型例子`)];

  return {
    ...outline,
    title: baseTopic,
    description:
      outline.language === 'en-US'
        ? `Explain ${baseTopic} line by line on the blackboard.`
        : `通过板书逐行讲解 ${baseTopic}。`,
    keyPoints,
    quizConfig: {
      questionCount: Math.min(
        MAX_BLACKBOARD_LINES,
        Math.max(3, outline.quizConfig?.questionCount ?? keyPoints.length),
      ),
      difficulty: outline.quizConfig?.difficulty ?? 'medium',
      questionTypes: ['single'],
    },
  };
}

export const maxDuration = 300;

/**
 * Incremental JSON array parser.
 * Extracts complete top-level objects from a partially-streamed JSON array.
 * Returns newly found objects (skipping `alreadyParsed` count).
 */
function extractNewOutlines(buffer: string, alreadyParsed: number): SceneOutline[] {
  const results: SceneOutline[] = [];

  // Find the start of the JSON array (skip any markdown fencing)
  const stripped = buffer.replace(/^[\s\S]*?(?=\[)/, '');
  const arrayStart = stripped.indexOf('[');
  if (arrayStart === -1) return results;

  let depth = 0;
  let objectStart = -1;
  let inString = false;
  let escaped = false;
  let objectCount = 0;

  for (let i = arrayStart + 1; i < stripped.length; i++) {
    const char = stripped[i];

    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === '\\' && inString) {
      escaped = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;

    if (char === '{') {
      if (depth === 0) objectStart = i;
      depth++;
    } else if (char === '}') {
      depth--;
      if (depth === 0 && objectStart >= 0) {
        objectCount++;
        if (objectCount > alreadyParsed) {
          try {
            const obj = JSON.parse(stripped.substring(objectStart, i + 1));
            results.push(obj);
          } catch {
            // Incomplete or invalid JSON — skip
          }
        }
        objectStart = -1;
      }
    }
  }

  return results;
}

function enforceMandatoryBlackboardImageAndVideo(
  outlines: SceneOutline[],
  language: string,
  imageEnabled: boolean,
  videoEnabled: boolean,
): SceneOutline[] {
  const next = outlines.map((outline) => normalizeQuizOutlineAsBlackboard(outline));

  const hasQuiz = next.some((o) => o.type === 'quiz');
  if (!hasQuiz) {
    next.push(
      normalizeQuizOutlineAsBlackboard({
        id: nanoid(),
        type: 'quiz',
        title: language === 'zh-CN' ? '核心概念' : 'Core Concept',
        description:
          language === 'zh-CN'
            ? '通过板书逐行讲解核心概念。'
            : 'Explain the core concept line by line on the blackboard.',
        keyPoints:
          language === 'zh-CN'
            ? ['核心概念定义', '推导步骤', '结论与要点']
            : ['Core concept definition', 'Derivation steps', 'Conclusion and key takeaway'],
        order: next.length + 1,
        language: language as SceneOutline['language'],
        quizConfig: {
          questionCount: 3,
          difficulty: 'medium',
          questionTypes: ['single'],
        },
      } as SceneOutline),
    );
  }

  enforceGeneratedImageBudget(next, language, imageEnabled);

  if (videoEnabled) {
    arrangeIntroAndClosingVideos(next, language);
    enforceGeneratedImageBudget(next, language, imageEnabled);
  }

  return next.map((o, i) => ({ ...o, order: i + 1 }));
}

function makeImagePromptForOutline(outline: SceneOutline, language: string, index: number): string {
  const points = (outline.keyPoints || []).slice(0, 3).join('; ');
  const topic = [outline.title, points].filter(Boolean).join(' - ');
  const languageHint =
    language === 'zh-CN'
      ? 'If any unavoidable text appears, use Simplified Chinese.'
      : 'If any unavoidable text appears, use English.';

  return [
    `A content-specific educational illustration for lesson scene ${index}: ${topic}.`,
    'Depict the actual concept, example, imagery, process, or historical context being taught.',
    'No classroom, no teacher, no students, no slide screenshot, no title banner, no poster layout, no readable labels or captions.',
    languageHint,
  ].join(' ');
}

function nextGeneratedImageId(usedIds: Set<string>): string {
  for (let i = 1; i <= MAX_AI_IMAGES_PER_COURSE; i += 1) {
    const id = `gen_img_${i}`;
    if (!usedIds.has(id)) return id;
  }
  let i = MAX_AI_IMAGES_PER_COURSE + 1;
  while (usedIds.has(`gen_img_${i}`)) i += 1;
  return `gen_img_${i}`;
}

function enforceGeneratedImageBudget(
  outlines: SceneOutline[],
  language: string,
  imageEnabled: boolean,
): void {
  const usedIds = new Set<string>();
  let imageCount = 0;

  for (const outline of outlines) {
    const media = outline.mediaGenerations || [];
    const kept = media.filter((mediaItem) => {
      if (mediaItem.type !== 'image') {
        if (mediaItem.elementId) usedIds.add(mediaItem.elementId);
        return true;
      }
      if (imageCount >= MAX_AI_IMAGES_PER_COURSE || !imageEnabled) return false;
      imageCount += 1;
      usedIds.add(mediaItem.elementId);
      return true;
    });
    outline.mediaGenerations = kept.length ? kept : undefined;
  }

  if (!imageEnabled || imageCount > 0 || outlines.length === 0) return;

  const targets = outlines.filter(
    (outline) =>
      (outline.type === 'slide' || outline.type === 'quiz') &&
      !outline.mediaGenerations?.some((media) => media.type === 'video'),
  );
  const usableTargets = targets.length > 0 ? targets : outlines;
  let targetIndex = 0;

  while (imageCount < 1 && usableTargets.length > 0) {
    const target = usableTargets[targetIndex % usableTargets.length];
    const elementId = nextGeneratedImageId(usedIds);
    usedIds.add(elementId);
    const mediaGenerations = target.mediaGenerations ? [...target.mediaGenerations] : [];
    mediaGenerations.push({
      type: 'image',
      elementId,
      aspectRatio: '16:9',
      prompt: makeImagePromptForOutline(target, language, imageCount + 1),
      style: 'educational illustration',
    });
    target.mediaGenerations = mediaGenerations;
    imageCount += 1;
    targetIndex += 1;
  }
}

function isSummaryOutline(outline: SceneOutline): boolean {
  const text = `${outline.title} ${outline.description} ${(outline.keyPoints || []).join(' ')}`;
  return /总结|回顾|收束|要点|小结|summary|recap|takeaway|conclusion/i.test(text);
}

function makeVideoPromptForOutline(
  outline: SceneOutline,
  language: string,
  purpose: 'intro' | 'closing',
): string {
  const points = (outline.keyPoints || []).slice(0, 4).join('; ');
  const topic = [outline.title, outline.description, points].filter(Boolean).join(' - ');
  const purposeText =
    purpose === 'intro'
      ? 'opening guide video that introduces the lesson context'
      : 'closing recap video that visually connects the lesson content before the final summary';

  return [
    `A 12-second ${purposeText}: ${topic}.`,
    'Visualize the actual concept, imagery, process, example, or historical/literary context being taught.',
    'Use cinematic educational visuals with natural ambient sound, no classroom, no teacher, no students, no slide screenshot, no poster layout, no readable captions.',
    language === 'zh-CN'
      ? 'If unavoidable text appears, use Simplified Chinese only.'
      : 'If unavoidable text appears, use English only.',
  ].join(' ');
}

function makeVideoRequest(
  language: string,
  elementId: string,
  outline: SceneOutline,
  purpose: 'intro' | 'closing',
) {
  return {
    type: 'video' as const,
    elementId,
    aspectRatio: '16:9' as const,
    duration: 12,
    resolution: '1080p' as const,
    generateAudio: true,
    prompt: makeVideoPromptForOutline(outline, language, purpose),
  };
}

function makeSummaryOutline(language: string): SceneOutline {
  return {
    id: nanoid(),
    type: 'slide',
    title: language === 'zh-CN' ? '课程总结' : 'Lesson Summary',
    description:
      language === 'zh-CN'
        ? '总结本节课的核心内容，帮助学生完成最后回顾。'
        : 'Summarize the lesson and close with key takeaways.',
    keyPoints:
      language === 'zh-CN'
        ? ['核心内容回顾', '关键理解方式', '课后思考方向']
        : ['Core content recap', 'Key understanding', 'Next reflection'],
    order: 999,
    language: language as SceneOutline['language'],
  };
}

function makeVideoOutline(language: string): SceneOutline {
  return {
    id: nanoid(),
    type: 'slide',
    title: language === 'zh-CN' ? '课程内容短片' : 'Lesson Visual Clip',
    description:
      language === 'zh-CN'
        ? '串场后直接播放短片，把本节课的内容用画面连起来。'
        : 'Play a short content-focused clip immediately after a transition line.',
    keyPoints:
      language === 'zh-CN'
        ? ['接下来用一段短片回看本节课的核心内容']
        : ['Next, watch a short clip that revisits the lesson content'],
    order: 998,
    language: language as SceneOutline['language'],
    mediaGenerations: [],
  };
}

function makeIntroVideoOutline(language: string, source?: SceneOutline): SceneOutline {
  const outline: SceneOutline = {
    id: source?.id || nanoid(),
    type: 'slide',
    title: language === 'zh-CN' ? '课程导读短片' : 'Lesson Opening Video',
    description:
      source?.description ||
      (language === 'zh-CN'
        ? '用一段导读短片进入课程情境。'
        : 'Use a short guide video to enter the lesson context.'),
    keyPoints:
      source?.keyPoints?.length
        ? source.keyPoints.slice(0, 3)
        : language === 'zh-CN'
          ? ['进入课程情境', '理解核心主题']
          : ['Enter the lesson context', 'Understand the core topic'],
    order: 1,
    language: language as SceneOutline['language'],
  };

  return {
    ...outline,
    mediaGenerations: [makeVideoRequest(language, 'gen_vid_1', outline, 'intro')],
  };
}

function arrangeIntroAndClosingVideos(outlines: SceneOutline[], language: string): void {
  let providedClosingVideo: NonNullable<SceneOutline['mediaGenerations']>[number] | undefined;

  for (let i = 0; i < outlines.length; i++) {
    const media = outlines[i].mediaGenerations || [];
    const video = media.find((m) => m.type === 'video');
    if (video && !providedClosingVideo) {
      providedClosingVideo = {
        ...video,
        type: 'video',
        elementId: 'gen_vid_2',
        aspectRatio: '16:9',
        duration: 12,
        resolution: '1080p',
        generateAudio: true,
      };
    }
    outlines[i] = {
      ...outlines[i],
      mediaGenerations: media.filter((m) => m.type !== 'video'),
    };
  }

  for (let i = outlines.length - 1; i >= 0; i--) {
    if (!outlines[i].mediaGenerations?.length) {
      delete outlines[i].mediaGenerations;
    }
  }

  let introSource: SceneOutline | undefined;
  const first = outlines[0];
  if (
    first &&
    first.type !== 'quiz' &&
    /导读|引入|开场|背景|intro|opening|overview|background/i.test(
      `${first.title} ${first.description}`,
    )
  ) {
    introSource = outlines.shift();
  }
  const introScene = makeIntroVideoOutline(language, introSource);

  let videoScene = makeVideoOutline(language);
  const videoRequest =
    providedClosingVideo || makeVideoRequest(language, 'gen_vid_2', videoScene, 'closing');
  videoScene = {
    ...videoScene,
    type: 'slide',
    title: language === 'zh-CN' ? '课程内容短片' : 'Lesson Visual Clip',
    description:
      language === 'zh-CN'
        ? '串场后直接播放短片，把本节课的内容用画面连起来。'
        : 'Play a short content-focused clip immediately after a transition line.',
    keyPoints:
      language === 'zh-CN'
        ? ['接下来用一段短片回看本节课的核心内容']
        : ['Next, watch a short clip that revisits the lesson content'],
    mediaGenerations: [
      {
        ...videoRequest,
        elementId: 'gen_vid_2',
        duration: 12,
        resolution: '1080p',
        generateAudio: true,
      },
    ],
  };

  let summaryScene: SceneOutline | undefined;
  const last = outlines[outlines.length - 1];
  if (last && isSummaryOutline(last)) {
    summaryScene = last;
    outlines.pop();
  } else {
    summaryScene = makeSummaryOutline(language);
  }

  outlines.unshift(introScene);
  outlines.push(videoScene, {
    ...summaryScene,
    type: 'slide',
    mediaGenerations: (summaryScene.mediaGenerations || []).filter((m) => m.type !== 'video'),
  });
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    // Get API configuration from request headers
    const { model: languageModel, modelInfo, modelString } = resolveModelFromHeaders(req);

    if (!body.requirements) {
      return apiError('MISSING_REQUIRED_FIELD', 400, 'Requirements are required');
    }

    const { requirements, pdfText, pdfImages, imageMapping, researchContext, agents } = body as {
      requirements: UserRequirements;
      pdfText?: string;
      pdfImages?: PdfImage[];
      imageMapping?: ImageMapping;
      researchContext?: string;
      agents?: AgentInfo[];
    };

    // Detect vision capability
    const hasVision = !!modelInfo?.capabilities?.vision;

    // Build prompt (same logic as generateSceneOutlinesFromRequirements)
    let availableImagesText =
      requirements.language === 'zh-CN' ? '无可用图片' : 'No images available';
    let visionImages: Array<{ id: string; src: string }> | undefined;

    if (pdfImages && pdfImages.length > 0) {
      if (hasVision && imageMapping) {
        // Vision mode: split into vision images (first N) and text-only (rest)
        const allWithSrc = pdfImages.filter((img) => imageMapping[img.id]);
        const visionSlice = allWithSrc.slice(0, MAX_VISION_IMAGES);
        const textOnlySlice = allWithSrc.slice(MAX_VISION_IMAGES);
        const noSrcImages = pdfImages.filter((img) => !imageMapping[img.id]);

        const visionDescriptions = visionSlice.map((img) =>
          formatImagePlaceholder(img, requirements.language),
        );
        const textDescriptions = [...textOnlySlice, ...noSrcImages].map((img) =>
          formatImageDescription(img, requirements.language),
        );
        availableImagesText = [...visionDescriptions, ...textDescriptions].join('\n');

        visionImages = visionSlice.map((img) => ({
          id: img.id,
          src: imageMapping[img.id],
          width: img.width,
          height: img.height,
        }));
      } else {
        // Text-only mode: full descriptions
        availableImagesText = pdfImages
          .map((img) => formatImageDescription(img, requirements.language))
          .join('\n');
      }
    }

    // Build media generation policy based on enabled flags
    const imageGenerationEnabled = req.headers.get('x-image-generation-enabled') !== 'false';
    const videoGenerationEnabled = req.headers.get('x-video-generation-enabled') !== 'false';
    let mediaGenerationPolicy = '';
    if (!imageGenerationEnabled && !videoGenerationEnabled) {
      mediaGenerationPolicy =
        '**IMPORTANT: Do NOT include any mediaGenerations in the outlines. Both image and video generation are disabled.**';
    } else if (!imageGenerationEnabled) {
      mediaGenerationPolicy =
        '**IMPORTANT: Do NOT include any image mediaGenerations (type: "image") in the outlines. Image generation is disabled. Video generation is allowed.**';
    } else if (!videoGenerationEnabled) {
      mediaGenerationPolicy =
        '**IMPORTANT: Do NOT include any video mediaGenerations (type: "video") in the outlines. Video generation is disabled. Image generation is allowed.**';
    }

    // Build teacher context from agents (if available)
    const teacherContext = formatTeacherPersonaForPrompt(agents);

    const prompts = buildPrompt(PROMPT_IDS.REQUIREMENTS_TO_OUTLINES, {
      requirement: requirements.requirement,
      language: requirements.language,
      pdfContent: pdfText
        ? pdfText.substring(0, MAX_PDF_CONTENT_CHARS)
        : requirements.language === 'zh-CN'
          ? '无'
          : 'None',
      availableImages: availableImagesText,
      researchContext: researchContext || (requirements.language === 'zh-CN' ? '无' : 'None'),
      mediaGenerationPolicy,
      teacherContext,
    });

    if (!prompts) {
      return apiError('INTERNAL_ERROR', 500, 'Prompt template not found');
    }

    log.info(
      `Generating outlines: "${requirements.requirement.substring(0, 50)}" [model=${modelString}]`,
    );

    // Create SSE stream with heartbeat to prevent connection timeout
    const encoder = new TextEncoder();
    const HEARTBEAT_INTERVAL_MS = 15_000;
    const stream = new ReadableStream({
      async start(controller) {
        // Heartbeat: periodically send SSE comments to keep the connection alive.
        let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
        const startHeartbeat = () => {
          stopHeartbeat();
          heartbeatTimer = setInterval(() => {
            try {
              controller.enqueue(encoder.encode(`:heartbeat\n\n`));
            } catch {
              stopHeartbeat();
            }
          }, HEARTBEAT_INTERVAL_MS);
        };
        const stopHeartbeat = () => {
          if (heartbeatTimer) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
          }
        };

        const MAX_STREAM_RETRIES = 2;

        try {
          startHeartbeat();

          const streamParams = visionImages?.length
            ? {
                model: languageModel,
                system: prompts.system,
                messages: [
                  {
                    role: 'user' as const,
                    content: buildVisionUserContent(prompts.user, visionImages),
                  },
                ],
                maxOutputTokens: modelInfo?.outputWindow,
              }
            : {
                model: languageModel,
                system: prompts.system,
                prompt: prompts.user,
                maxOutputTokens: modelInfo?.outputWindow,
              };

          let parsedOutlines: SceneOutline[] = [];
          let lastError: string | undefined;

          for (let attempt = 1; attempt <= MAX_STREAM_RETRIES + 1; attempt++) {
            try {
              const result = streamLLM(streamParams, 'scene-outlines-stream');

              let fullText = '';
              parsedOutlines = [];

              for await (const chunk of result.textStream) {
                fullText += chunk;

                // Try to extract new outlines from the accumulated text
                const newOutlines = extractNewOutlines(fullText, parsedOutlines.length);
                for (const outline of newOutlines) {
                  // Ensure ID and order
                  const enriched = {
                    ...outline,
                    id: outline.id || nanoid(),
                    order: parsedOutlines.length + 1,
                  };
                  parsedOutlines.push(enriched);

                  const event = JSON.stringify({
                    type: 'outline',
                    data: enriched,
                    index: parsedOutlines.length - 1,
                  });
                  controller.enqueue(encoder.encode(`data: ${event}\n\n`));
                }
              }

              // Validate: got outlines?
              if (parsedOutlines.length > 0) break;

              // Empty result — retry if we have attempts left
              lastError = fullText.trim()
                ? 'LLM response could not be parsed into outlines'
                : 'LLM returned empty response';

              if (attempt <= MAX_STREAM_RETRIES) {
                log.warn(
                  `Empty outlines (attempt ${attempt}/${MAX_STREAM_RETRIES + 1}), retrying...`,
                );
                // Notify client a retry is happening
                const retryEvent = JSON.stringify({
                  type: 'retry',
                  attempt,
                  maxAttempts: MAX_STREAM_RETRIES + 1,
                });
                controller.enqueue(encoder.encode(`data: ${retryEvent}\n\n`));
              }
            } catch (error) {
              lastError = error instanceof Error ? error.message : String(error);

              if (attempt <= MAX_STREAM_RETRIES) {
                log.warn(
                  `Stream error (attempt ${attempt}/${MAX_STREAM_RETRIES + 1}), retrying...`,
                  error,
                );
                const retryEvent = JSON.stringify({
                  type: 'retry',
                  attempt,
                  maxAttempts: MAX_STREAM_RETRIES + 1,
                });
                controller.enqueue(encoder.encode(`data: ${retryEvent}\n\n`));
                continue;
              }
            }
          }

          if (parsedOutlines.length > 0) {
            // Replace sequential gen_img_N/gen_vid_N with globally unique IDs
            const enforced = enforceMandatoryBlackboardImageAndVideo(
              parsedOutlines,
              requirements.language,
              imageGenerationEnabled,
              videoGenerationEnabled,
            );
            const grounded = ensureContentGroundedMediaPrompts(
              enforced,
              requirements.requirement,
              requirements.language,
            );
            const uniquifiedOutlines = uniquifyMediaElementIds(grounded);
            // Send done event with all outlines
            const doneEvent = JSON.stringify({
              type: 'done',
              outlines: uniquifiedOutlines,
            });
            controller.enqueue(encoder.encode(`data: ${doneEvent}\n\n`));
          } else {
            // All retries exhausted, no outlines produced
            log.error(
              `Outline generation failed after ${MAX_STREAM_RETRIES + 1} attempts: ${lastError}`,
            );
            const errorEvent = JSON.stringify({
              type: 'error',
              error: lastError || 'Failed to generate outlines',
            });
            controller.enqueue(encoder.encode(`data: ${errorEvent}\n\n`));
          }
        } catch (error) {
          const errorEvent = JSON.stringify({
            type: 'error',
            error: error instanceof Error ? error.message : String(error),
          });
          controller.enqueue(encoder.encode(`data: ${errorEvent}\n\n`));
        } finally {
          stopHeartbeat();
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    });
  } catch (error) {
    log.error('Streaming error:', error);
    return apiError('INTERNAL_ERROR', 500, error instanceof Error ? error.message : String(error));
  }
}
