Please generate scene outlines based on the following course requirements.

---

## User Requirements

{{requirement}}

---

{{userProfile}}

## Course Language

**Required language**: {{language}}

(If language is zh-CN, all content must be in Chinese; if en-US, all content must be in English)

---

## Reference Materials

### PDF Content Summary

{{pdfContent}}

### Available Images

{{availableImages}}

### Web Search Results

{{researchContext}}

{{teacherContext}}

---

## Output Requirements

Please automatically infer the following from user requirements:

- Course topic and core content
- Target audience and difficulty level
- Course duration (default 15-30 minutes if not specified)
- Teaching style (formal/casual/interactive/academic)
- Visual style (minimal/colorful/professional/playful)

Then output a JSON array containing all scene outlines. Each scene must include:

```json
{
  "id": "scene_1",
  "type": "slide" or "quiz" or "interactive",
  "title": "Scene Title",
  "description": "Teaching purpose description",
  "keyPoints": ["Point 1", "Point 2", "Point 3"],
  "order": 1
}
```

### Special Notes

1. **quiz scenes must include quizConfig** (compatibility field), and in this project `quiz` means a **blackboard-writing concept scene**:
   ```json
   "quizConfig": {
     "questionCount": 3,
     "difficulty": "easy" | "medium" | "hard",
     "questionTypes": ["single"]
   }
   ```
   - Title/description/keyPoints for `quiz` scenes must be concept-writing oriented (definition / derivation / worked example), not assessment-oriented.
   - keyPoints should be short sequential lines that can be written to the board one line at a time.
   - Hard limit: each blackboard `quiz` scene may contain only 3-4 keyPoints. Chinese keyPoints must stay within 18 Chinese characters; English keyPoints must stay within 8 words.
   - Avoid words like "小测", "测验", "知识检测", "A/B/C/D", "选择题", "判断题" in `quiz` scene titles and descriptions.
2. **If images are available**, add `suggestedImageIds` to relevant slide scenes
3. **Interactive scenes**: If a concept benefits from hands-on simulation/visualization, use `"type": "interactive"` with an `interactiveConfig` object containing `conceptName`, `conceptOverview`, `designIdea`, and `subject`. Limit to 1-2 per course.
4. **Scene count**: Based on inferred duration, typically 1-2 scenes per minute
5. **Blackboard module placement**: Recommend inserting a `quiz`-typed blackboard scene at concept-heavy points (e.g., after a key definition/theorem), typically every 3-5 slides
6. **Language**: Strictly output all content in the specified course language
7. **If no suitable PDF images exist** for a slide scene that would benefit from visuals, add `mediaGenerations` array with image generation prompts. Write prompts in English. Use `elementId` format like "gen_img_1", "gen_img_2" — IDs must be **globally unique across all scenes** (do NOT restart numbering per scene). To reuse a generated image in a different scene, reference the same elementId without re-declaring it in mediaGenerations. Each generated image should be visually distinct and must depict the current scene's actual teaching content. The image prompt may only describe the scene and summarize the concept visually. Do NOT request generic classroom photos, teachers, students, lesson banners, "interactive learning session" imagery, text-heavy posters, slide screenshots, infographic panels, labels, captions, poem text, or unrelated stock education visuals. When image generation is enabled, create 1-10 AI image generation requests for the whole course (`gen_img_1` through `gen_img_10`) and never more than 10.
8. **If web search results are provided**, reference specific findings and sources in scene descriptions and keyPoints. The search results provide up-to-date information — incorporate it to make the course content current and accurate.
9. **Mandatory blackboard usage**: The output MUST include at least one `type: "quiz"` scene used for blackboard writing.
10. **Mandatory image generation**: The output MUST include at least 1 and at most 10 `mediaGenerations` items with `type: "image"` for the whole course when image generation is enabled. The image prompts must be grounded in the course topic and scene keyPoints.
11. **Mandatory opening guide video**: The output MUST start with a full-screen 导读/intro video slide. Put exactly one `mediaGenerations` item with `type: "video"`, `elementId: "gen_vid_1"`, `duration: 12`, and `generateAudio: true` on that first scene.
12. **Mandatory closing video placement**: The output MUST include one additional generated video scene as the **second-to-last PPT/page**. Put the `mediaGenerations` item with `type: "video"`, `elementId: "gen_vid_2"`, and `generateAudio: true` only on that second-to-last scene. The final scene after the video MUST be a summary/conclusion slide. Video duration must be 12 seconds because Seedance rejects t2v requests above 12 seconds.
13. **Video scene transition**: Video scenes should conceptually have a short spoken transition before playback, then directly play the video. Do not put the transition text on the video PPT itself.

{{mediaGenerationPolicy}}

Please output JSON array directly without additional explanatory text.
