'use client';

import { useEffect, useState, Suspense, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'motion/react';
import { CheckCircle2, Sparkles, AlertCircle, AlertTriangle, ArrowLeft, Bot } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { useStageStore } from '@/lib/store/stage';
import { useSettingsStore } from '@/lib/store/settings';
import { useAgentRegistry } from '@/lib/orchestration/registry/store';
import { getAvailableProvidersWithVoices } from '@/lib/audio/voice-resolver';
import { useI18n } from '@/lib/hooks/use-i18n';
import {
  loadImageMapping,
  loadPdfBlob,
  cleanupOldImages,
  storeImages,
} from '@/lib/utils/image-storage';
import { getCurrentModelConfig } from '@/lib/utils/model-config';
import { db } from '@/lib/utils/database';
import { MAX_PDF_CONTENT_CHARS, MAX_VISION_IMAGES } from '@/lib/constants/generation';
import { nanoid } from 'nanoid';
import type { Stage } from '@/lib/types/stage';
import type { SceneOutline, PdfImage, ImageMapping } from '@/lib/types/generation';
import { AgentRevealModal } from '@/components/agent/agent-reveal-modal';
import { createLogger } from '@/lib/logger';
import { type GenerationSessionState, ALL_STEPS, getActiveSteps } from './types';
import { StepVisualizer } from './components/visualizers';

const log = createLogger('GenerationPreview');

// 声明一个真实课程架构类 ycf
type CourseBlueprint = {
  courseTitle: string;
  targetAudience: string;
  totalDurationMinutes: number;
  teachingStyle: string;
  modules: Array<{
    id: string;
    title: string;
    objective: string;
    sceneCountHint: number;
    recommendedSceneTypes: Array<'slide' | 'quiz' | 'interactive' | 'pbl'>;
  }>;
  assessmentPlan?: string;
};


function GenerationPreviewContent() {
  const router = useRouter();
  const { t } = useI18n();
  const hasStartedRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [session, setSession] = useState<GenerationSessionState | null>(null);
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isComplete] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [streamingOutlines, setStreamingOutlines] = useState<SceneOutline[] | null>(null);
  const [truncationWarnings, setTruncationWarnings] = useState<string[]>([]);
  const [webSearchSources, setWebSearchSources] = useState<Array<{ title: string; url: string }>>(
    [],
  );
  const [showAgentReveal, setShowAgentReveal] = useState(false);
  const [generatedAgents, setGeneratedAgents] = useState<
    Array<{
      id: string;
      name: string;
      role: string;
      persona: string;
      avatar: string;
      color: string;
      priority: number;
    }>
  >([]);
  const agentRevealResolveRef = useRef<(() => void) | null>(null);

  // 定义一个真实课程类ycf
  const [generatedBlueprint, setGeneratedBlueprint] = useState<CourseBlueprint | null>(null);


  // 定义两个手动放行位
  // 是否正在等待你手动放行
  const [waitingForOutlineConfirm, setWaitingForOutlineConfirm] = useState(false);

  // 用来保存“继续执行”的 resolve 函数
  const outlineConfirmResolveRef = useRef<(() => void) | null>(null);

  // Compute active steps based on session state
  const activeSteps = getActiveSteps(session);

  // Load session from sessionStorage
  useEffect(() => {
    cleanupOldImages(24).catch((e) => log.error(e));

    const saved = sessionStorage.getItem('generationSession');
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as GenerationSessionState;
        setSession(parsed);
      } catch (e) {
        log.error('Failed to parse generation session:', e);
      }
    }
    setSessionLoaded(true);
  }, []);

  // Abort all in-flight requests on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  // Get API credentials from localStorage
  const getApiHeaders = () => {
    const modelConfig = getCurrentModelConfig();
    const settings = useSettingsStore.getState();
    const imageProviderConfig = settings.imageProvidersConfig?.[settings.imageProviderId];
    const videoProviderConfig = settings.videoProvidersConfig?.[settings.videoProviderId];
    return {
      'Content-Type': 'application/json',
      'x-model': modelConfig.modelString,
      'x-api-key': modelConfig.apiKey,
      'x-base-url': modelConfig.baseUrl,
      'x-provider-type': modelConfig.providerType || '',
      'x-requires-api-key': modelConfig.requiresApiKey ? 'true' : 'false',
      // Image generation provider
      'x-image-provider': settings.imageProviderId || '',
      'x-image-model': settings.imageModelId || '',
      'x-image-api-key': imageProviderConfig?.apiKey || '',
      'x-image-base-url': imageProviderConfig?.baseUrl || '',
      // Video generation provider
      'x-video-provider': settings.videoProviderId || '',
      'x-video-model': settings.videoModelId || '',
      'x-video-api-key': videoProviderConfig?.apiKey || '',
      'x-video-base-url': videoProviderConfig?.baseUrl || '',
      // Media generation toggles
      'x-image-generation-enabled': String(settings.imageGenerationEnabled ?? true),
      'x-video-generation-enabled': String(settings.videoGenerationEnabled ?? true),
    };
  };

  // Auto-start generation when session is loaded
  useEffect(() => {
    if (session && !hasStartedRef.current) {
      hasStartedRef.current = true;
      startGeneration();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session]);

  // Main generation flow
  // startGeneration 是一个异步函数
  // 作用：启动整套“生成课程”的流程
  const startGeneration = async () => {
    // 如果 session 不存在，直接结束
    // session 可以理解成“本次生成任务的全部上下文”
    if (!session) return;

    // ---------------------------
    // 1. 准备本次生成任务的中断控制器
    // ---------------------------

    // 如果之前已经有一个还没结束的生成任务，先把它中断掉
    abortControllerRef.current?.abort();

    // 新建一个 AbortController
    // 作用：后面 fetch 请求都可以挂到它上面，如果需要取消，就能统一取消
    const controller = new AbortController();

    // 把新的 controller 保存起来
    abortControllerRef.current = controller;

    // signal 是“中断信号”
    // 后面 fetch(..., { signal }) 会用到它
    const signal = controller.signal;

    // ---------------------------
    // 2. 准备一个“本地可修改”的 session 副本
    // ---------------------------

    // currentSession 一开始等于 session
    // 后面如果 PDF 被解析了、web search 完成了，
    // currentSession 会不断更新
    let currentSession = session;

    // 清空之前的错误
    setError(null);

    // 把当前步骤索引先重置为 0
    setCurrentStepIndex(0);

    try {
      // ---------------------------
      // 3. 根据当前 session 计算“当前有哪些步骤”
      // ---------------------------

      // activeSteps = 这次生成真正需要执行的步骤列表
      // 比如有没有 PDF、有没有 webSearch，都会影响步骤列表
      let activeSteps = getActiveSteps(currentSession);

      // ---------------------------
      // 4. 判断是否需要先解析 PDF
      // ---------------------------

      // hasPdfToAnalyze 的意思：
      // - currentSession.pdfStorageKey 存在：说明有 PDF
      // - currentSession.pdfText 还没有：说明还没解析过
      const hasPdfToAnalyze = !!currentSession.pdfStorageKey && !currentSession.pdfText;

      // 如果这次不需要解析 PDF
      if (!hasPdfToAnalyze) {
        // 就去 activeSteps 里找第一个“不是 pdf-analysis”的步骤
        const firstNonPdfIdx = activeSteps.findIndex((s) => s.id !== 'pdf-analysis');

        // 把当前步骤切到那个步骤
        setCurrentStepIndex(Math.max(0, firstNonPdfIdx));
      }

      // ---------------------------
      // 5. Step 0: 如果需要，就先解析 PDF
      // ---------------------------

      if (hasPdfToAnalyze) {
        // 打日志：开始解析 PDF
        log.debug('=== Generation Preview: Parsing PDF ===');

        // 根据 pdfStorageKey 去加载 PDF blob
        const pdfBlob = await loadPdfBlob(currentSession.pdfStorageKey!);

        // 如果没加载到，报错
        if (!pdfBlob) {
          throw new Error(t('generation.pdfLoadFailed'));
        }

        // 再检查一次：这个 pdfBlob 必须真的是 Blob，而且大小不能为 0
        if (!(pdfBlob instanceof Blob) || pdfBlob.size === 0) {
          log.error('Invalid PDF blob:', {
            type: typeof pdfBlob,
            size: pdfBlob instanceof Blob ? pdfBlob.size : 'N/A',
          });
          throw new Error(t('generation.pdfLoadFailed'));
        }

        // 把 Blob 包装成 File 对象
        // 这样后面发 multipart/form-data 时更稳
        const pdfFile = new File([pdfBlob], currentSession.pdfFileName || 'document.pdf', {
          type: 'application/pdf',
        });

        // 新建一个 FormData
        // 这是上传文件常用的数据格式
        const parseFormData = new FormData();

        // 把 pdfFile 挂到字段 pdf 上
        parseFormData.append('pdf', pdfFile);

        // 如果有 PDF providerId，也一起传给后端
        if (currentSession.pdfProviderId) {
          parseFormData.append('providerId', currentSession.pdfProviderId);
        }

        // 如果有 pdf provider 的 apiKey，也传过去
        if (currentSession.pdfProviderConfig?.apiKey?.trim()) {
          parseFormData.append('apiKey', currentSession.pdfProviderConfig.apiKey);
        }

        // 如果有 baseUrl，也传过去
        if (currentSession.pdfProviderConfig?.baseUrl?.trim()) {
          parseFormData.append('baseUrl', currentSession.pdfProviderConfig.baseUrl);
        }

        // 请求后端接口 /api/parse-pdf，让后端解析 PDF
        const parseResponse = await fetch('/api/parse-pdf', {
          method: 'POST',
          body: parseFormData,
          signal,
        });

        // 如果后端返回失败
        if (!parseResponse.ok) {
          const errorData = await parseResponse.json();
          throw new Error(errorData.error || t('generation.pdfParseFailed'));
        }

        // 解析接口返回的数据
        const parseResult = await parseResponse.json();

        // 如果 success 不为真，或者 data 不存在，也报错
        if (!parseResult.success || !parseResult.data) {
          throw new Error(t('generation.pdfParseFailed'));
        }

        // 取出 PDF 里的文本
        let pdfText = parseResult.data.text as string;

        // 如果文字太长，截断
        if (pdfText.length > MAX_PDF_CONTENT_CHARS) {
          pdfText = pdfText.substring(0, MAX_PDF_CONTENT_CHARS);
        }

        // ---------------------------
        // 6. 整理 PDF 图片
        // ---------------------------

        // 优先从 metadata.pdfImages 取图像信息
        const rawPdfImages = parseResult.data.metadata?.pdfImages;

        // 如果 metadata.pdfImages 存在，就按它来构建 images
        // 否则退回到 parseResult.data.images 这个旧格式
        const images = rawPdfImages
          ? rawPdfImages.map(
              (img: {
                id: string;
                src?: string;
                pageNumber?: number;
                description?: string;
                width?: number;
                height?: number;
              }) => ({
                id: img.id,
                src: img.src || '',
                pageNumber: img.pageNumber || 1,
                description: img.description,
                width: img.width,
                height: img.height,
              }),
            )
          : (parseResult.data.images as string[]).map((src: string, i: number) => ({
              id: `img_${i + 1}`,
              src,
              pageNumber: 1,
            }));

        // 把这些图片存起来，得到每张图对应的 storageId
        const imageStorageIds = await storeImages(images);

        // 再把图片整理成 PdfImage[] 这种标准结构
        const pdfImages: PdfImage[] = images.map(
          (
            img: {
              id: string;
              src: string;
              pageNumber: number;
              description?: string;
              width?: number;
              height?: number;
            },
            i: number,
          ) => ({
            id: img.id,
            src: '',
            pageNumber: img.pageNumber,
            description: img.description,
            width: img.width,
            height: img.height,
            storageId: imageStorageIds[i],
          }),
        );

        // ---------------------------
        // 7. 用解析后的 PDF 数据更新 session
        // ---------------------------

        const updatedSession = {
          ...currentSession,
          pdfText,
          pdfImages,
          imageStorageIds,
          pdfStorageKey: undefined, // 清掉 storageKey，避免以后重复解析
        };

        // 更新 React 状态里的 session
        setSession(updatedSession);

        // 也同步存进 sessionStorage
        sessionStorage.setItem('generationSession', JSON.stringify(updatedSession));

        // ---------------------------
        // 8. 生成“截断警告”
        // ---------------------------

        const warnings: string[] = [];

        // 如果原始文本长度超限，就加一个文字截断警告
        if ((parseResult.data.text as string).length > MAX_PDF_CONTENT_CHARS) {
          warnings.push(
            t('generation.textTruncated').replace('{n}', String(MAX_PDF_CONTENT_CHARS)),
          );
        }

        // 如果图片太多，也加一个图片截断警告
        if (images.length > MAX_VISION_IMAGES) {
          warnings.push(
            t('generation.imageTruncated')
              .replace('{total}', String(images.length))
              .replace('{max}', String(MAX_VISION_IMAGES)),
          );
        }

        // 如果 warnings 里真的有警告，就存起来
        if (warnings.length > 0) {
          setTruncationWarnings(warnings);
        }

        // 用更新后的 session 覆盖当前局部变量
        currentSession = updatedSession;

        // 重新计算 activeSteps，因为 session 变了
        activeSteps = getActiveSteps(currentSession);
      }

      // ---------------------------
      // 9. Web Search 步骤（如果启用了联网搜索）
      // ---------------------------

      // 找到 web-search 这个步骤在 activeSteps 里的位置
      const webSearchStepIdx = activeSteps.findIndex((s) => s.id === 'web-search');

      // 如果用户要求了 webSearch，并且这个步骤存在
      if (currentSession.requirements.webSearch && webSearchStepIdx >= 0) {
        // 当前步骤切换到 web search
        setCurrentStepIndex(webSearchStepIdx);

        // 先清空上一次的搜索来源
        setWebSearchSources([]);

        // 读取 web search 相关设置
        const wsSettings = useSettingsStore.getState();

        // 读取 web search provider 对应的 apiKey
        const wsApiKey =
          wsSettings.webSearchProvidersConfig?.[wsSettings.webSearchProviderId]?.apiKey;

        // 调后端接口 /api/web-search
        const res = await fetch('/api/web-search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: currentSession.requirements.requirement,
            apiKey: wsApiKey || undefined,
          }),
          signal,
        });

        // 如果搜索失败，报错
        if (!res.ok) {
          const data = await res.json().catch(() => ({ error: 'Web search failed' }));
          throw new Error(data.error || t('generation.webSearchFailed'));
        }

        // 读取搜索结果
        const searchData = await res.json();

        // 整理 sources 数组，只保留 title 和 url
        const sources = (searchData.sources || []).map((s: { title: string; url: string }) => ({
          title: s.title,
          url: s.url,
        }));

        // 更新前端状态，显示这些搜索来源
        setWebSearchSources(sources);

        // 把搜索上下文和来源写回 session
        const updatedSessionWithSearch = {
          ...currentSession,
          researchContext: searchData.context || '',
          researchSources: sources,
        };

        setSession(updatedSessionWithSearch);
        sessionStorage.setItem('generationSession', JSON.stringify(updatedSessionWithSearch));

        // 更新局部变量
        currentSession = updatedSessionWithSearch;
        activeSteps = getActiveSteps(currentSession);
      }

      // ---------------------------
      // 10. 准备 imageMapping
      // ---------------------------

      // imageMapping 可以理解成：
      // “图片 ID 和真实图片数据/存储位置”的对应表
      let imageMapping: ImageMapping = {};

      // 如果 currentSession 里有 imageStorageIds
      if (currentSession.imageStorageIds && currentSession.imageStorageIds.length > 0) {
        log.debug('Loading images from IndexedDB');

        // 从 IndexedDB 里把这些图片映射关系读出来
        imageMapping = await loadImageMapping(currentSession.imageStorageIds);
      } else if (
        currentSession.imageMapping &&
        Object.keys(currentSession.imageMapping).length > 0
      ) {
        // 否则，如果 session 里已经有旧格式的 imageMapping，就直接用
        log.debug('Using imageMapping from session (old format)');
        imageMapping = currentSession.imageMapping;
      }

      // ---------------------------
      // 11. Agent generation
      // ---------------------------

      const settings = useSettingsStore.getState();

      // agents 数组，后面会存课堂中的角色
      let agents: Array<{
        id: string;
        name: string;
        role: string;
        persona?: string;
      }> = [];

      // 先生成一个新的 stageId
      const stageId = nanoid(10);

      // 再构造一个 stage 对象
      const stage: Stage = {
        id: stageId,
        name: extractTopicFromRequirement(currentSession.requirements.requirement),
        description: '',
        language: currentSession.requirements.language || 'zh-CN',
        style: 'professional',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      // 如果 agentMode 是 auto，说明角色由系统自动生成
      if (settings.agentMode === 'auto') {
        // 找到 agent-generation 这一步
        const agentStepIdx = activeSteps.findIndex((s) => s.id === 'agent-generation');

        // 如果找到了，就把当前步骤切到这里
        if (agentStepIdx >= 0) setCurrentStepIndex(agentStepIdx);

        try {
          // allAvatars = 可选头像池
          // 每个对象都包含：
          // - path: 头像图片路径
          // - desc: 头像描述
          const allAvatars = [
            { path: '/avatars/teacher.png', desc: 'Male teacher with glasses, holding a book, green background' },
            { path: '/avatars/teacher-2.png', desc: 'Female teacher with long dark hair, blue traditional outfit, gentle expression' },
            { path: '/avatars/assist.png', desc: 'Young female assistant with glasses, pink background, friendly smile' },
            { path: '/avatars/assist-2.png', desc: 'Young female in orange top and purple overalls, cheerful and approachable' },
            { path: '/avatars/clown.png', desc: 'Energetic girl with glasses pointing up, green shirt, lively and fun' },
            { path: '/avatars/clown-2.png', desc: 'Playful girl with curly hair doing rock gesture, blue shirt, humorous vibe' },
            { path: '/avatars/curious.png', desc: 'Surprised boy with glasses, hand on cheek, curious expression' },
            { path: '/avatars/curious-2.png', desc: 'Boy with backpack holding a book and question mark bubble, inquisitive' },
            { path: '/avatars/note-taker.png', desc: 'Studious boy with glasses, blue shirt, calm and organized' },
            { path: '/avatars/note-taker-2.png', desc: 'Active boy with yellow backpack waving, blue outfit, enthusiastic learner' },
            { path: '/avatars/thinker.png', desc: 'Thoughtful girl with hand on chin, purple background, contemplative' },
            { path: '/avatars/thinker-2.png', desc: 'Girl reading a book intently, long dark hair, intellectual and focused' },
          ];

          // 定义一个函数：收集所有可用语音
          const getAvailableVoicesForGeneration = () => {
            const providers = getAvailableProvidersWithVoices(settings.ttsProvidersConfig);

            return providers.flatMap((p) =>
              p.voices.map((v) => ({
                providerId: p.providerId,
                voiceId: v.id,
                voiceName: v.name,
              })),
            );
          };

          // 调后端接口生成 agent profiles
          const agentResp = await fetch('/api/generate/agent-profiles', {
            method: 'POST',
            headers: getApiHeaders(),
            body: JSON.stringify({
              stageInfo: { name: stage.name, description: stage.description },
              language: currentSession.requirements.language || 'zh-CN',
              availableAvatars: allAvatars.map((a) => a.path),
              avatarDescriptions: allAvatars.map((a) => ({ path: a.path, desc: a.desc })),
              availableVoices: getAvailableVoicesForGeneration(),
            }),
            signal,
          });

          // 如果 agent 生成失败，直接报错
          if (!agentResp.ok) throw new Error('Agent generation failed');

          const agentData = await agentResp.json();

          if (!agentData.success) throw new Error(agentData.error || 'Agent generation failed');

          // 把生成出的 agents 存入注册表 / 数据库
          const { saveGeneratedAgents } = await import('@/lib/orchestration/registry/store');
          const savedIds = await saveGeneratedAgents(stage.id, agentData.agents);

          // 更新设置：选中这些新生成的 agents
          settings.setSelectedAgentIds(savedIds);

          // stage 记录这些 agent ids
          stage.agentIds = savedIds;

          // 显示“角色揭晓”弹窗
          setGeneratedAgents(agentData.agents);
          setShowAgentReveal(true);

          // 等待用户看完揭晓动画
          await new Promise<void>((resolve) => {
            agentRevealResolveRef.current = resolve;
          });

          // 把 registry 里的 agent 真正整理成 agents 数组
          agents = savedIds
            .map((id) => useAgentRegistry.getState().getAgent(id))
            .filter(Boolean)
            .map((a) => ({
              id: a!.id,
              name: a!.name,
              role: a!.role,
              persona: a!.persona,
            }));
        } catch (err: unknown) {
          // 如果自动生成 agent 失败，就退回到预设 agent
          log.warn('[Generation] Agent generation failed, falling back to presets:', err);

          const registry = useAgentRegistry.getState();

          const fallbackIds = settings.selectedAgentIds.filter((id) => {
            const a = registry.getAgent(id);
            return a && !a.isGenerated;
          });

          agents = fallbackIds
            .map((id) => registry.getAgent(id))
            .filter(Boolean)
            .map((a) => ({
              id: a!.id,
              name: a!.name,
              role: a!.role,
              persona: a!.persona,
            }));

          stage.agentIds = fallbackIds;
        }
      } else {
        // 如果不是 auto，而是 preset mode，就直接使用已选定的 preset agents
        const registry = useAgentRegistry.getState();

        const presetAgentIds = settings.selectedAgentIds.filter((id) => {
          const a = registry.getAgent(id);
          return a && !a.isGenerated;
        });

        agents = presetAgentIds
          .map((id) => registry.getAgent(id))
          .filter(Boolean)
          .map((a) => ({
            id: a!.id,
            name: a!.name,
            role: a!.role,
            persona: a!.persona,
          }));

        stage.agentIds = presetAgentIds;
      }

      // ── Generate course blueprint (before outlines) ──
      // 生成顶层课程结构ycf
      // 先读开关：只有勾选了，才启用真实课程架构
      // ── Generate course blueprint (temporary visual version) ──

      // 先看 session 里有没有打开“真实课程架构”开关
      const useStructureGuide = !!currentSession.structureGuideEnabled;

      // 只有开关打开时，才处理 blueprint
      if (useStructureGuide) {
        let blueprint = currentSession.courseBlueprint as CourseBlueprint | null;

        // 如果还没有 blueprint，就先生成一份临时假数据
        if (!blueprint) {
          // 让页面上先显示一条状态文字
          setStatusMessage('正在生成顶层课程结构...');

          // 这里先用“前端临时假数据”做可视化
          // 第三步你把后端接口写完后，再把这一块替换成 fetch('/api/generate/course-blueprint')
          blueprint = {
            courseTitle: extractTopicFromRequirement(currentSession.requirements.requirement),
            targetAudience:
              currentSession.requirements.userBio || currentSession.requirements.userNickname
                ? `${currentSession.requirements.userNickname || '学习者'}`
                : '普通学习者',
            totalDurationMinutes: 20,
            teachingStyle: 'interactive',
            modules: [
              {
                id: 'module_1',
                title: '课程导入与目标建立',
                objective: '帮助学生理解课程主题、背景和学习目标',
                sceneCountHint: 2,
                recommendedSceneTypes: ['slide'],
              },
              {
                id: 'module_2',
                title: '核心概念讲解',
                objective: '系统讲解本课程最关键的概念和知识结构',
                sceneCountHint: 4,
                recommendedSceneTypes: ['slide', 'interactive'],
              },
              {
                id: 'module_3',
                title: '理解检查与应用',
                objective: '通过练习或测验检查理解情况，并引导应用',
                sceneCountHint: 2,
                recommendedSceneTypes: ['quiz', 'slide'],
              },
              {
                id: 'module_4',
                title: '总结与迁移',
                objective: '总结核心要点，并提示学生如何迁移到新问题',
                sceneCountHint: 1,
                recommendedSceneTypes: ['slide'],
              },
            ],
            assessmentPlan: '在核心概念模块后插入一次知识检查，并在结尾进行总结回顾',
          };

          // 把 blueprint 放到页面状态里，这样页面就能显示出来
          setGeneratedBlueprint(blueprint);

          // 同时写回 session，后面 outlines 也能继续用
          const updatedSessionWithBlueprint = {
            ...currentSession,
            courseBlueprint: blueprint,
          };

          setSession(updatedSessionWithBlueprint);
          sessionStorage.setItem(
            'generationSession',
            JSON.stringify(updatedSessionWithBlueprint),
          );
          currentSession = updatedSessionWithBlueprint;
        } else {
          // 如果 session 里本来就有 blueprint，就直接拿来显示
          setGeneratedBlueprint(blueprint);
        }
      } else {
        // 没勾选真实课程架构，就不显示 blueprint
        setGeneratedBlueprint(null);
      }


      // // ---------------------------
      // // 11.5 在 Generate outlines 前手动阻塞，等待按钮确认
      // // ---------------------------

      // setStatusMessage('等待你确认后，再开始生成课纲...');
      // setWaitingForOutlineConfirm(true);

      // await new Promise<void>((resolve) => {
      //   outlineConfirmResolveRef.current = resolve;
      // });

      // setWaitingForOutlineConfirm(false);
      // outlineConfirmResolveRef.current = null;

      // ---------------------------
      // 12. Generate outlines
      // ---------------------------

      // outlines = 当前 session 里已有的 sceneOutlines
      let outlines = currentSession.sceneOutlines;

      // 找到 outline 步骤的位置
      const outlineStepIdx = activeSteps.findIndex((s) => s.id === 'outline');

      // 当前步骤切到 outline
      setCurrentStepIndex(outlineStepIdx >= 0 ? outlineStepIdx : 0);

      // 如果还没有 outlines，就去生成
      if (!outlines || outlines.length === 0) {
        log.debug('=== Generating outlines (SSE) ===');

        // streamingOutlines 先清空
        setStreamingOutlines([]);

        // 用 Promise 包一层，等待 SSE 把 outlines 一条条流回来
        outlines = await new Promise<SceneOutline[]>((resolve, reject) => {
          const collected: SceneOutline[] = [];
          // 生成课纲
          // requirements需求
          // pdfText,pdfImages PDF信息
          // imageMapping图片映射   researchContext联网搜索上下文
          fetch('/api/generate/scene-outlines-stream', {
            method: 'POST',
            headers: getApiHeaders(),
            body: JSON.stringify({
              requirements: currentSession.requirements,
              pdfText: currentSession.pdfText,
              pdfImages: currentSession.pdfImages,
              imageMapping,
              researchContext: currentSession.researchContext,
              agents,
            }),
            signal,
          })
            .then((res) => {
              if (!res.ok) {
                return res.json().then((d) => {
                  reject(new Error(d.error || t('generation.outlineGenerateFailed')));
                });
              }

              // 从响应体中拿 reader
              const reader = res.body?.getReader();

              if (!reader) {
                reject(new Error(t('generation.streamNotReadable')));
                return;
              }

              // 用 TextDecoder 把二进制流转成文本
              const decoder = new TextDecoder();

              // sseBuffer = 暂存尚未处理完的 SSE 文本
              let sseBuffer = '';

              // pump = 不断读取 SSE 流
              const pump = (): Promise<void> =>
                reader.read().then(({ done, value }) => {
                  if (value) {
                    // 把新读到的数据解码并拼到 buffer
                    sseBuffer += decoder.decode(value, { stream: !done });

                    // 按换行切分
                    const lines = sseBuffer.split('\n');

                    // 最后一段可能不完整，暂存起来
                    sseBuffer = lines.pop() || '';

                    for (const line of lines) {
                      // SSE 的有效数据一般以 "data: " 开头
                      if (!line.startsWith('data: ')) continue;

                      try {
                        // 去掉 "data: " 前缀，再解析 JSON
                        const evt = JSON.parse(line.slice(6));

                        // 如果这是 outline 事件
                        if (evt.type === 'outline') {
                          collected.push(evt.data);
                          setStreamingOutlines([...collected]);
                        }
                        // 如果后端让你 retry
                        else if (evt.type === 'retry') {
                          collected.length = 0;
                          setStreamingOutlines([]);
                          setStatusMessage(t('generation.outlineRetrying'));
                        }
                        // 如果 done 了，说明 outlines 全部到齐
                        else if (evt.type === 'done') {
                          resolve(evt.outlines || collected);
                          return;
                        }
                        // 如果 error 了，直接 reject
                        else if (evt.type === 'error') {
                          reject(new Error(evt.error));
                          return;
                        }
                      } catch (e) {
                        log.error('Failed to parse outline SSE:', line, e);
                      }
                    }
                  }

                  // 如果流结束了
                  if (done) {
                    if (collected.length > 0) {
                      resolve(collected);
                    } else {
                      reject(new Error(t('generation.outlineEmptyResponse')));
                    }
                    return;
                  }

                  // 继续读下一段
                  return pump();
                });

              pump().catch(reject);
            })
            .catch(reject);
        });

        // outlines 生成完后，把它写回 session
        const updatedSession = { ...currentSession, sceneOutlines: outlines };
        setSession(updatedSession);
        sessionStorage.setItem('generationSession', JSON.stringify(updatedSession));

        // outlines 成功后，清掉首页 requirementDraft
        try {
          localStorage.removeItem('requirementDraft');
        } catch {
          /* ignore */
        }

        // 停 800ms，让用户看一眼提纲完成状态
        await new Promise((resolve) => setTimeout(resolve, 800));
      }

      // ---------------------------
      // 13. outlines 检查 + 存到 stage store
      // ---------------------------

      setStatusMessage('');

      // 如果 outlines 还是空，报错
      if (!outlines || outlines.length === 0) {
        throw new Error(t('generation.outlineEmptyResponse'));
      }

      // 取 stage store
      const store = useStageStore.getState();

      // 把 stage 写进去
      store.setStage(stage);

      // 把 outlines 写进去
      store.setOutlines(outlines);

      // 找到 slide-content 步骤
      const contentStepIdx = activeSteps.findIndex((s) => s.id === 'slide-content');

      // 当前步骤切过去
      if (contentStepIdx >= 0) setCurrentStepIndex(contentStepIdx);

      // ---------------------------
      // 14. 准备生成第一页内容需要的上下文
      // ---------------------------

      const stageInfo = {
        name: stage.name,
        description: stage.description,
        language: stage.language,
        style: stage.style,
      };

      // userProfile = 把用户昵称 / 简介拼成一段字符串
      const userProfile =
        currentSession.requirements.userNickname || currentSession.requirements.userBio
          ? `Student: ${currentSession.requirements.userNickname || 'Unknown'}${currentSession.requirements.userBio ? ` — ${currentSession.requirements.userBio}` : ''}`
          : undefined;

      // 告诉 store：这些 outlines 正在生成中
      store.setGeneratingOutlines(outlines);

      // 这里只生成第一个场景
      const firstOutline = outlines[0];

      // ---------------------------
      // 15. 生成第一个场景的内容
      // ---------------------------
      const contentResp = await fetch('/api/generate/scene-content', {
        method: 'POST',
        headers: getApiHeaders(),
        body: JSON.stringify({
          outline: firstOutline,
          allOutlines: outlines,
          pdfImages: currentSession.pdfImages,
          imageMapping,
          stageInfo,
          stageId: stage.id,
          agents,
        }),
        signal,
      });

      // 如果后端返回失败
      if (!contentResp.ok) {
        const errorData = await contentResp.json().catch(() => ({ error: 'Request failed' }));
        throw new Error(errorData.error || t('generation.sceneGenerateFailed'));
      }

      // 解析内容生成结果
      const contentData = await contentResp.json();

      // 如果 success 为假，或者 content 不存在，就报错
      if (!contentData.success || !contentData.content) {
        throw new Error(contentData.error || t('generation.sceneGenerateFailed'));
      }

      // ---------------------------
      // 16. 根据内容生成动作
      // ---------------------------

      // 找到 actions 步骤
      const actionsStepIdx = activeSteps.findIndex((s) => s.id === 'actions');

      // 当前步骤切换到 actions
      setCurrentStepIndex(actionsStepIdx >= 0 ? actionsStepIdx : currentStepIndex + 1);

      const actionsResp = await fetch('/api/generate/scene-actions', {
        method: 'POST',
        headers: getApiHeaders(),
        body: JSON.stringify({
          outline: contentData.effectiveOutline || firstOutline,
          allOutlines: outlines,
          content: contentData.content,
          stageId: stage.id,
          agents,
          previousSpeeches: [],
          userProfile,
        }),
        signal,
      });

      // 如果动作生成失败
      if (!actionsResp.ok) {
        const errorData = await actionsResp.json().catch(() => ({ error: 'Request failed' }));
        throw new Error(errorData.error || t('generation.sceneGenerateFailed'));
      }

      const data = await actionsResp.json();

      // 如果 scene 没生成出来，也报错
      if (!data.success || !data.scene) {
        throw new Error(data.error || t('generation.sceneGenerateFailed'));
      }

      // ---------------------------
      // 17. 生成 TTS
      // ---------------------------

      // 如果启用了 TTS，并且不是浏览器原生 TTS
      if (settings.ttsEnabled && settings.ttsProviderId !== 'browser-native-tts') {
        // 读取当前 TTS provider 配置
        const ttsProviderConfig = settings.ttsProvidersConfig?.[settings.ttsProviderId];

        // 从 data.scene.actions 里筛出所有 speech 动作
        const speechActions = (data.scene.actions || []).filter(
          (a: { type: string; text?: string }) => a.type === 'speech' && a.text,
        );

        let ttsFailCount = 0;

        // 逐个 speech action 生成语音
        for (const action of speechActions) {
          const audioId = `tts_${action.id}`;

          // 把 audioId 挂回 action
          action.audioId = audioId;

          try {
            // 调后端接口 /api/generate/tts
            const resp = await fetch('/api/generate/tts', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                text: action.text,
                audioId,
                ttsProviderId: settings.ttsProviderId,
                ttsVoice: settings.ttsVoice,
                ttsSpeed: settings.ttsSpeed,
                ttsApiKey: ttsProviderConfig?.apiKey || undefined,
                ttsBaseUrl: ttsProviderConfig?.baseUrl || undefined,
              }),
              signal,
            });

            // 如果失败，就累计失败次数
            if (!resp.ok) {
              ttsFailCount++;
              continue;
            }

            const ttsData = await resp.json();

            if (!ttsData.success) {
              ttsFailCount++;
              continue;
            }

            // 后端返回的是 base64 音频，把它解码成二进制
            const binary = atob(ttsData.base64);

            const bytes = new Uint8Array(binary.length);

            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

            // 再封装成 Blob
            const blob = new Blob([bytes], { type: `audio/${ttsData.format}` });

            // 保存到 IndexedDB
            await db.audioFiles.put({
              id: audioId,
              blob,
              format: ttsData.format,
              createdAt: Date.now(),
            });
          } catch (err) {
            log.warn(`[TTS] Failed for ${audioId}:`, err);
            ttsFailCount++;
          }
        }

        // 如果有 speech action，且有任何 TTS 失败，就报错
        if (ttsFailCount > 0 && speechActions.length > 0) {
          throw new Error(t('generation.speechFailed'));
        }
      }


      // ---------------------------
      // 18. 把生成好的第一页 scene 放入 store，并跳转 classroom
      // ---------------------------

      // 把 scene 放进 store
      store.addScene(data.scene);

      // 当前场景设置成这个 scene
      store.setCurrentSceneId(data.scene.id);

      // 剩下还没生成的 outlines，作为 skeleton placeholders 先挂着
      const remaining = outlines.filter((o) => o.order !== data.scene.order);
      store.setGeneratingOutlines(remaining);

      // 把 classroom 页面后续继续生成需要的参数存起来
      sessionStorage.setItem(
        'generationParams',
        JSON.stringify({
          pdfImages: currentSession.pdfImages,
          agents,
          userProfile,
        }),
      );

      // 清掉 generationSession
      sessionStorage.removeItem('generationSession');

      // 把 store 保存到持久化存储
      await store.saveToStorage();

      // 最后跳转到 classroom 页面
      router.push(`/classroom/${stage.id}`);
    } catch (err) {
      // ---------------------------
      // 19. 错误处理
      // ---------------------------

      // 如果是 AbortError，说明这是正常中断，不当作错误弹给用户
      if (err instanceof DOMException && err.name === 'AbortError') {
        log.info('[GenerationPreview] Generation aborted');
        return;
      }

      // 真正失败时，清掉 generationSession
      sessionStorage.removeItem('generationSession');

      // 把错误显示到前端状态里
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const extractTopicFromRequirement = (requirement: string): string => {
    const trimmed = requirement.trim();
    if (trimmed.length <= 500) {
      return trimmed;
    }
    return trimmed.substring(0, 500).trim() + '...';
  };

  const goBackToHome = () => {
    abortControllerRef.current?.abort();
    sessionStorage.removeItem('generationSession');
    router.push('/');
  };

  // Still loading session from sessionStorage
  if (!sessionLoaded) {
    return (
      <div className="min-h-[100dvh] w-full bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex items-center justify-center p-4">
        <div className="text-center text-muted-foreground">
          <div className="size-8 border-2 border-current border-t-transparent rounded-full animate-spin mx-auto" />
        </div>
      </div>
    );
  }

  // No session found
  if (!session) {
    return (
      <div className="min-h-[100dvh] w-full bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex items-center justify-center p-4">
        <Card className="p-8 max-w-md w-full">
          <div className="text-center space-y-4">
            <AlertCircle className="size-12 text-muted-foreground mx-auto" />
            <h2 className="text-xl font-semibold">{t('generation.sessionNotFound')}</h2>
            <p className="text-sm text-muted-foreground">{t('generation.sessionNotFoundDesc')}</p>
            <Button onClick={() => router.push('/')} className="w-full">
              <ArrowLeft className="size-4 mr-2" />
              {t('generation.backToHome')}
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  const activeStep =
    activeSteps.length > 0
      ? activeSteps[Math.min(currentStepIndex, activeSteps.length - 1)]
      : ALL_STEPS[0];

  return (
    <div className="min-h-[100dvh] w-full bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex flex-col items-center justify-center p-4 relative overflow-hidden text-center">
      {/* Background Decor */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div
          className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDuration: '4s' }}
        />
        <div
          className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDuration: '6s' }}
        />
      </div>

      {/* Back button */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-4 left-4 z-20"
      >
        <Button variant="ghost" size="sm" onClick={goBackToHome}>
          <ArrowLeft className="size-4 mr-2" />
          {t('generation.backToHome')}
        </Button>
      </motion.div>

      <div className="z-10 w-full max-w-lg space-y-8 flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full"
        >
          <Card className="relative overflow-hidden border-muted/40 shadow-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl min-h-[400px] flex flex-col items-center justify-center p-8 md:p-12">
            {/* Progress Dots */}
            <div className="absolute top-6 left-0 right-0 flex justify-center gap-2">
              {activeSteps.map((step, idx) => (
                <div
                  key={step.id}
                  className={cn(
                    'h-1.5 rounded-full transition-all duration-500',
                    idx < currentStepIndex
                      ? 'w-1.5 bg-blue-500/30'
                      : idx === currentStepIndex
                        ? 'w-8 bg-blue-500'
                        : 'w-1.5 bg-muted/50',
                  )}
                />
              ))}
            </div>

            {/* Central Content */}
            <div className="flex-1 flex flex-col items-center justify-center w-full space-y-8 mt-4">
              {/* Icon / Visualizer Container */}
              <div className="relative size-48 flex items-center justify-center">
                <AnimatePresence mode="popLayout">
                  {error ? (
                    <motion.div
                      key="error"
                      initial={{ scale: 0.5, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className="size-32 rounded-full bg-red-500/10 flex items-center justify-center border-2 border-red-500/20"
                    >
                      <AlertCircle className="size-16 text-red-500" />
                    </motion.div>
                  ) : isComplete ? (
                    <motion.div
                      key="complete"
                      initial={{ scale: 0.5, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className="size-32 rounded-full bg-green-500/10 flex items-center justify-center border-2 border-green-500/20"
                    >
                      <CheckCircle2 className="size-16 text-green-500" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key={activeStep.id}
                      initial={{ scale: 0.8, opacity: 0, filter: 'blur(10px)' }}
                      animate={{ scale: 1, opacity: 1, filter: 'blur(0px)' }}
                      exit={{ scale: 1.2, opacity: 0, filter: 'blur(10px)' }}
                      transition={{ duration: 0.4 }}
                      className="absolute inset-0 flex items-center justify-center"
                    >
                      <StepVisualizer
                        stepId={activeStep.id}
                        outlines={streamingOutlines}
                        webSearchSources={webSearchSources}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Text Content */}
              <div className="space-y-3 max-w-sm mx-auto">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={error ? 'error' : isComplete ? 'done' : activeStep.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-2"
                  >
                    <h2 className="text-2xl font-bold tracking-tight">
                      {error
                        ? t('generation.generationFailed')
                        : isComplete
                          ? t('generation.generationComplete')
                          : t(activeStep.title)}
                    </h2>
                    <p className="text-muted-foreground text-base">
                      {error
                        ? error
                        : isComplete
                          ? t('generation.classroomReady')
                          : statusMessage || t(activeStep.description)}
                    </p>
                  </motion.div>
                </AnimatePresence>

                {/* Truncation warning indicator */}
                <AnimatePresence>
                  {truncationWarnings.length > 0 && !error && !isComplete && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0 }}
                      transition={{
                        type: 'spring',
                        stiffness: 500,
                        damping: 30,
                      }}
                      className="flex justify-center"
                    >
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <motion.button
                            type="button"
                            animate={{
                              boxShadow: [
                                '0 0 0 0 rgba(251, 191, 36, 0), 0 0 0 0 rgba(251, 191, 36, 0)',
                                '0 0 16px 4px rgba(251, 191, 36, 0.12), 0 0 4px 1px rgba(251, 191, 36, 0.08)',
                                '0 0 0 0 rgba(251, 191, 36, 0), 0 0 0 0 rgba(251, 191, 36, 0)',
                              ],
                            }}
                            transition={{
                              duration: 3,
                              repeat: Infinity,
                              ease: 'easeInOut',
                            }}
                            className="relative size-7 rounded-full flex items-center justify-center cursor-default
                                       bg-gradient-to-br from-amber-400/15 to-orange-400/10
                                       border border-amber-400/25 hover:border-amber-400/40
                                       hover:from-amber-400/20 hover:to-orange-400/15
                                       transition-colors duration-300
                                       focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/30"
                          >
                            <AlertTriangle
                              className="size-3.5 text-amber-500 dark:text-amber-400"
                              strokeWidth={2.5}
                            />
                          </motion.button>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" sideOffset={6}>
                          <div className="space-y-1 py-0.5">
                            {truncationWarnings.map((w, i) => (
                              <p key={i} className="text-xs leading-relaxed">
                                {w}
                              </p>
                            ))}
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              {/* ycf可视化 */}
              {/* Blueprint Preview - 第二步临时可视化 */}
              {generatedBlueprint && !error && (
                <div className="w-full max-w-xl mx-auto mt-6 rounded-2xl border border-blue-200/60 bg-white/70 dark:bg-slate-900/60 backdrop-blur p-4 text-left shadow-sm">
                  <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                    顶层课程结构（Blueprint 预览）
                  </div>

                  <div className="mt-2 text-xs text-muted-foreground">
                    <span className="font-medium">课程标题：</span>
                    {generatedBlueprint.courseTitle}
                  </div>

                  <div className="mt-1 text-xs text-muted-foreground">
                    <span className="font-medium">目标受众：</span>
                    {generatedBlueprint.targetAudience}
                  </div>

                  <div className="mt-1 text-xs text-muted-foreground">
                    <span className="font-medium">总时长：</span>
                    {generatedBlueprint.totalDurationMinutes} 分钟
                    <span className="ml-3 font-medium">教学风格：</span>
                    {generatedBlueprint.teachingStyle}
                  </div>

                  <div className="mt-4 space-y-3">
                    {generatedBlueprint.modules.map((module, index) => (
                      <div
                        key={module.id}
                        className="rounded-xl border border-slate-200/70 dark:border-slate-700/70 bg-background/60 p-3"
                      >
                        <div className="text-sm font-medium">
                          {index + 1}. {module.title}
                        </div>

                        <div className="mt-1 text-xs text-muted-foreground">
                          {module.objective}
                        </div>

                        <div className="mt-2 text-[11px] text-muted-foreground">
                          建议场景数：{module.sceneCountHint}
                          <span className="mx-2">·</span>
                          推荐类型：{module.recommendedSceneTypes.join(' / ')}
                        </div>
                      </div>
                    ))}
                  </div>

                  {generatedBlueprint.assessmentPlan && (
                    <div className="mt-4 rounded-lg bg-blue-50 dark:bg-blue-950/30 px-3 py-2 text-xs text-blue-700 dark:text-blue-300">
                      <span className="font-medium">评估设计：</span>
                      {generatedBlueprint.assessmentPlan}
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>
        </motion.div>

        {/* Footer Action */}
        <div className="h-16 flex items-center justify-center w-full">
          <AnimatePresence>
            {error ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full max-w-xs"
              >
                <Button size="lg" variant="outline" className="w-full h-12" onClick={goBackToHome}>
                  {t('generation.goBackAndRetry')}
                </Button>
              </motion.div>
            ) : !isComplete ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-3 text-sm text-muted-foreground/50 font-medium uppercase tracking-widest"
              >
                <Sparkles className="size-3 animate-pulse" />
                {t('generation.aiWorking')}
                {generatedAgents.length > 0 && !showAgentReveal && (
                  <button
                    onClick={() => setShowAgentReveal(true)}
                    className="ml-2 flex items-center gap-1.5 rounded-full border border-purple-300/30 bg-purple-500/10 px-3 py-1 text-xs font-medium normal-case tracking-normal text-purple-400 transition-colors hover:bg-purple-500/20 hover:text-purple-300"
                  >
                    <Bot className="size-3" />
                    {t('generation.viewAgents')}
                  </button>
                )}
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>
      </div>

      {/* Agent Reveal Modal */}
      <AgentRevealModal
        agents={generatedAgents}
        open={showAgentReveal}
        onClose={() => setShowAgentReveal(false)}
        onAllRevealed={() => {
          agentRevealResolveRef.current?.();
          agentRevealResolveRef.current = null;
        }}
      />
    </div>
  );
}

export default function GenerationPreviewPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-[100dvh] w-full bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex items-center justify-center">
          <div className="animate-pulse space-y-4 text-center">
            <div className="h-8 w-48 bg-muted rounded mx-auto" />
            <div className="h-4 w-64 bg-muted rounded mx-auto" />
          </div>
        </div>
      }
    >
      <GenerationPreviewContent />
    </Suspense>
  );
}
