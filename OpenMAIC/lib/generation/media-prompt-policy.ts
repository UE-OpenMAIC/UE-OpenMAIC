import type { SceneOutline } from '@/lib/types/generation';

const GENERIC_CLASSROOM_PROMPT_PATTERNS = [
  /interactive learning session/i,
  /classroom/i,
  /teacher/i,
  /students?/i,
  /lecture/i,
  /whiteboard/i,
  /learning session/i,
  /school class/i,
  /presentation/i,
  /title banner/i,
  /text-heavy/i,
  /infographic/i,
];

function compact(text: string | undefined, maxLength: number): string {
  return (text || '')
    .replace(/\s+/g, ' ')
    .replace(/["<>]/g, '')
    .trim()
    .slice(0, maxLength);
}

function summarizeScene(outline: SceneOutline): string {
  const pieces = [
    outline.title,
    outline.description,
    ...(outline.keyPoints || []).slice(0, 5),
  ].filter(Boolean);
  return compact(pieces.join('; '), 420);
}

function isGenericClassroomPrompt(prompt: string): boolean {
  return GENERIC_CLASSROOM_PROMPT_PATTERNS.some((pattern) => pattern.test(prompt));
}

function buildImagePrompt(outline: SceneOutline, courseRequirement: string, language: string): string {
  const sceneSummary = summarizeScene(outline);
  const courseContext = compact(courseRequirement, 220);

  return [
    `Scene description only: ${sceneSummary}.`,
    `Short course summary: ${courseContext}.`,
    'Generate a single visual scene that summarizes the lesson content through setting, objects, mood, symbols, and composition.',
    'No classroom, no teacher, no students, no lecture, no presentation slide, no infographic layout, no title banner, no poster design.',
    'No readable text, no poem text, no labels, no captions, no large typography. The image should be a scene illustration, not a text card.',
    'Safe academic style, clean composition, content-specific visual metaphor.',
  ].join(' ');
}

function buildVideoPrompt(outline: SceneOutline, courseRequirement: string, language: string): string {
  const sceneSummary = summarizeScene(outline);
  const courseContext = compact(courseRequirement, 220);
  const textRule =
    language === 'zh-CN'
      ? 'Any visible labels or captions should be in Simplified Chinese.'
      : 'Any visible labels or captions should be in English.';

  return [
    `A 12-second video scene description: ${sceneSummary}.`,
    `Short course summary: ${courseContext}.`,
    'Visualize the lesson content through setting, objects, mood, symbols, and gentle camera movement.',
    'No classroom footage, no teacher, no students, no lecture, no presentation slide, no title banner.',
    'Avoid readable text; use visual storytelling and topic-specific imagery instead.',
    textRule,
    'Safe academic tone, clear focal subject, polished educational style.',
  ].join(' ');
}

export function ensureContentGroundedMediaPrompts(
  outlines: SceneOutline[],
  courseRequirement: string,
  language: string,
): SceneOutline[] {
  return outlines.map((outline) => {
    if (!outline.mediaGenerations?.length) return outline;

    return {
      ...outline,
      mediaGenerations: outline.mediaGenerations.map((media) => {
        if (media.type === 'image') {
          const generatedPrompt = buildImagePrompt(outline, courseRequirement, language);
          return {
            ...media,
            prompt: generatedPrompt,
          };
        }

        if (media.type === 'video') {
          const generatedPrompt = buildVideoPrompt(outline, courseRequirement, language);
          const prompt = media.prompt || '';
          return {
            ...media,
            prompt: isGenericClassroomPrompt(prompt)
              ? generatedPrompt
              : `${generatedPrompt} Original requested motion intent: ${compact(prompt, 260)}.`,
          };
        }

        return media;
      }),
    };
  });
}
