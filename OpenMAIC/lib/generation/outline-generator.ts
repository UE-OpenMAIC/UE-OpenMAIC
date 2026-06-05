/**
 * Stage 1: Generate scene outlines from user requirements.
 * Also contains outline fallback logic.
 */

import { nanoid } from 'nanoid';
import { MAX_PDF_CONTENT_CHARS, MAX_VISION_IMAGES } from '@/lib/constants/generation';
import type {
  UserRequirements,
  SceneOutline,
  PdfImage,
  ImageMapping,
} from '@/lib/types/generation';
import { buildPrompt, PROMPT_IDS } from './prompts';
import { formatImageDescription, formatImagePlaceholder } from './prompt-formatters';
import { parseJsonResponse } from './json-repair';
import { uniquifyMediaElementIds } from './scene-builder';
import { ensureContentGroundedMediaPrompts } from './media-prompt-policy';
import type { AICallFn, GenerationResult, GenerationCallbacks } from './pipeline-types';
import { createLogger } from '@/lib/logger';
const log = createLogger('Generation');
const MAX_AI_IMAGES_PER_COURSE = 10;
const MAX_BLACKBOARD_LINES = 4;

function trimBlackboardLine(line: string): string {
  const compact = line.replace(/\s+/g, ' ').trim();
  const hasCjk = /[\u3400-\u9fff]/.test(compact);
  const maxLength = hasCjk ? 18 : 58;
  return compact.length > maxLength ? `${compact.slice(0, maxLength - 1)}…` : compact;
}
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
      : [`${baseTopic} 的定义`, `${baseTopic} 的判定条件`, `${baseTopic} 的典型例子`];

  return {
    ...outline,
    title: `${baseTopic}板书推导`,
    description: `通过板书逐行讲解 ${baseTopic}，强调定义、步骤和结论。`,
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

function ensureMandatoryBlackboardImageAndVideo(
  outlines: SceneOutline[],
  language: string,
  imageEnabled: boolean,
  videoEnabled: boolean,
): SceneOutline[] {
  const next = [...outlines];

  const hasQuiz = next.some((o) => o.type === 'quiz');
  if (!hasQuiz) {
    const order = next.length + 1;
    next.push(
      normalizeQuizOutlineAsBlackboard({
        id: nanoid(),
        type: 'quiz',
        title: language === 'zh-CN' ? '核心概念板书推导' : 'Core Concept Board Writing',
        description:
          language === 'zh-CN'
            ? '通过板书逐行讲解核心概念，强调定义、步骤和结论。'
            : 'Explain the core concept line by line on the blackboard, emphasizing definition, steps, and conclusion.',
        keyPoints:
          language === 'zh-CN'
            ? ['核心概念定义', '推导步骤', '结论与要点']
            : ['Core concept definition', 'Derivation steps', 'Conclusion and key takeaway'],
        order,
        language,
        quizConfig: {
          questionCount: 3,
          difficulty: 'medium',
          questionTypes: ['single'],
        },
      } as SceneOutline),
    );
  }

  enforceGeneratedImageBudget(next, language, imageEnabled);

  if (videoEnabled && next.length > 0) {
    arrangeIntroAndClosingVideos(next, language);
    enforceGeneratedImageBudget(next, language, imageEnabled);
  }

  return next.map((outline, idx) => ({
    ...outline,
    order: idx + 1,
  }));
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
        : 'Summarize the core content of the lesson and close with key takeaways.',
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
        ? '用一段短片把本节课的核心意象和内容串联起来。'
        : 'Use a short clip to connect the lesson content before the final summary.',
    keyPoints:
      language === 'zh-CN'
        ? ['观看短片，回顾核心内容', '把画面与刚才的知识点联系起来']
        : ['Watch the clip to revisit the core content', 'Connect the visuals to the lesson points'],
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

  const withoutEmptyMedia = outlines.map((outline) => ({
    ...outline,
    mediaGenerations: outline.mediaGenerations?.length ? outline.mediaGenerations : undefined,
  }));
  outlines.splice(0, outlines.length, ...withoutEmptyMedia);

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
    title: summaryScene.title || (language === 'zh-CN' ? '课程总结' : 'Lesson Summary'),
    mediaGenerations: (summaryScene.mediaGenerations || []).filter((m) => m.type !== 'video'),
  });
}

/**
 * Generate scene outlines from user requirements
 * Now uses simplified UserRequirements with just requirement text and language
 */
export async function generateSceneOutlinesFromRequirements(
  requirements: UserRequirements,
  pdfText: string | undefined,
  pdfImages: PdfImage[] | undefined,
  aiCall: AICallFn,
  callbacks?: GenerationCallbacks,
  options?: {
    visionEnabled?: boolean;
    imageMapping?: ImageMapping;
    imageGenerationEnabled?: boolean;
    videoGenerationEnabled?: boolean;
    researchContext?: string;
    teacherContext?: string;
  },
): Promise<GenerationResult<SceneOutline[]>> {
  // Build available images description for the prompt
  let availableImagesText =
    requirements.language === 'zh-CN' ? '无可用图片' : 'No images available';
  let visionImages: Array<{ id: string; src: string }> | undefined;

  if (pdfImages && pdfImages.length > 0) {
    if (options?.visionEnabled && options?.imageMapping) {
      // Vision mode: split into vision images (first N) and text-only (rest)
      const allWithSrc = pdfImages.filter((img) => options.imageMapping![img.id]);
      const visionSlice = allWithSrc.slice(0, MAX_VISION_IMAGES);
      const textOnlySlice = allWithSrc.slice(MAX_VISION_IMAGES);
      const noSrcImages = pdfImages.filter((img) => !options.imageMapping![img.id]);

      const visionDescriptions = visionSlice.map((img) =>
        formatImagePlaceholder(img, requirements.language),
      );
      const textDescriptions = [...textOnlySlice, ...noSrcImages].map((img) =>
        formatImageDescription(img, requirements.language),
      );
      availableImagesText = [...visionDescriptions, ...textDescriptions].join('\n');

      visionImages = visionSlice.map((img) => ({
        id: img.id,
        src: options.imageMapping![img.id],
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

  // Build user profile string for prompt injection
  const userProfileText =
    requirements.userNickname || requirements.userBio
      ? `## Student Profile\n\nStudent: ${requirements.userNickname || 'Unknown'}${requirements.userBio ? ` — ${requirements.userBio}` : ''}\n\nConsider this student's background when designing the course. Adapt difficulty, examples, and teaching approach accordingly.\n\n---`
      : '';

  // Build media generation policy based on enabled flags
  const imageEnabled = options?.imageGenerationEnabled ?? true;
  const videoEnabled = options?.videoGenerationEnabled ?? true;
  let mediaGenerationPolicy = '';
  if (!imageEnabled && !videoEnabled) {
    mediaGenerationPolicy =
      '**IMPORTANT: Do NOT include any mediaGenerations in the outlines. Both image and video generation are disabled.**';
  } else if (!imageEnabled) {
    mediaGenerationPolicy =
      '**IMPORTANT: Do NOT include any image mediaGenerations (type: "image") in the outlines. Image generation is disabled. Video generation is allowed.**';
  } else if (!videoEnabled) {
    mediaGenerationPolicy =
      '**IMPORTANT: Do NOT include any video mediaGenerations (type: "video") in the outlines. Video generation is disabled. Image generation is allowed.**';
  }

  // Use simplified prompt variables
  const prompts = buildPrompt(PROMPT_IDS.REQUIREMENTS_TO_OUTLINES, {
    // New simplified variables
    requirement: requirements.requirement,
    language: requirements.language,
    pdfContent: pdfText
      ? pdfText.substring(0, MAX_PDF_CONTENT_CHARS)
      : requirements.language === 'zh-CN'
        ? '无'
        : 'None',
    availableImages: availableImagesText,
    userProfile: userProfileText,
    mediaGenerationPolicy,
    researchContext:
      options?.researchContext || (requirements.language === 'zh-CN' ? '无' : 'None'),
    // Server-side generation populates this via options; client-side populates via formatTeacherPersonaForPrompt
    teacherContext: options?.teacherContext || '',
  });

  if (!prompts) {
    return { success: false, error: 'Prompt template not found' };
  }

  try {
    callbacks?.onProgress?.({
      currentStage: 1,
      overallProgress: 20,
      stageProgress: 50,
      statusMessage: '正在分析需求，生成场景大纲...',
      scenesGenerated: 0,
      totalScenes: 0,
    });

    const response = await aiCall(prompts.system, prompts.user, visionImages);
    const outlines = parseJsonResponse<SceneOutline[]>(response);

    if (!outlines || !Array.isArray(outlines)) {
      return {
        success: false,
        error: 'Failed to parse scene outlines response',
      };
    }
    // Ensure IDs, order, and language
    const enriched = outlines.map((outline, index) =>
      normalizeQuizOutlineAsBlackboard({
        ...outline,
        id: outline.id || nanoid(),
        order: index + 1,
        language: requirements.language,
      }),
    );

    const enforced = ensureMandatoryBlackboardImageAndVideo(
      enriched,
      requirements.language,
      imageEnabled,
      videoEnabled,
    );
    const grounded = ensureContentGroundedMediaPrompts(
      enforced,
      requirements.requirement,
      requirements.language,
    );

    // Replace sequential gen_img_N/gen_vid_N with globally unique IDs
    const result = uniquifyMediaElementIds(grounded);

    callbacks?.onProgress?.({
      currentStage: 1,
      overallProgress: 50,
      stageProgress: 100,
      statusMessage: `已生成 ${result.length} 个场景大纲`,
      scenesGenerated: 0,
      totalScenes: result.length,
    });

    return { success: true, data: result };
  } catch (error) {
    return { success: false, error: String(error) };
  }
}

/**
 * Apply type fallbacks for outlines that can't be generated as their declared type.
 * - interactive without interactiveConfig → slide
 * - pbl without pblConfig or languageModel → slide
 */
export function applyOutlineFallbacks(
  outline: SceneOutline,
  hasLanguageModel: boolean,
): SceneOutline {
  outline = normalizeQuizOutlineAsBlackboard(outline);

  if (outline.type === 'interactive' && !outline.interactiveConfig) {
    log.warn(
      `Interactive outline "${outline.title}" missing interactiveConfig, falling back to slide`,
    );
    return { ...outline, type: 'slide' };
  }
  if (outline.type === 'pbl' && (!outline.pblConfig || !hasLanguageModel)) {
    log.warn(
      `PBL outline "${outline.title}" missing pblConfig or languageModel, falling back to slide`,
    );
    return { ...outline, type: 'slide' };
  }
  return outline;
}
