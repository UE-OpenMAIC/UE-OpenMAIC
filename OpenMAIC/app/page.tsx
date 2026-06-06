'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'motion/react';
import {
  ArrowUp,
  Check,
  ChevronDown,
  Clock,
  Copy,
  ImagePlus,
  Pencil,
  Trash2,
  Settings,
  Sun,
  Moon,
  Monitor,
  BotOff,
  ChevronUp,
} from 'lucide-react';
import { useI18n } from '@/lib/hooks/use-i18n';
import { createLogger } from '@/lib/logger';
import { Button } from '@/components/ui/button';
import { Textarea as UITextarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { SettingsDialog } from '@/components/settings';
import { GenerationToolbar } from '@/components/generation/generation-toolbar';
import { AgentBar } from '@/components/agent/agent-bar';
import { useTheme } from '@/lib/hooks/use-theme';
import { nanoid } from 'nanoid';
import { storePdfBlob } from '@/lib/utils/image-storage';
import type { UserRequirements } from '@/lib/types/generation';
import { useSettingsStore } from '@/lib/store/settings';
import { useUserProfileStore, AVATAR_OPTIONS } from '@/lib/store/user-profile';
import {
  StageListItem,
  listStages,
  deleteStageData,
  getFirstSlideByStages,
} from '@/lib/utils/stage-storage';
import { ThumbnailSlide } from '@/components/slide-renderer/components/ThumbnailSlide';
import type { Slide } from '@/lib/types/slides';
import { useMediaGenerationStore } from '@/lib/store/media-generation';
import { toast } from 'sonner';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useDraftCache } from '@/lib/hooks/use-draft-cache';
import { SpeechButton } from '@/components/audio/speech-button';

const log = createLogger('Home');

const WEB_SEARCH_STORAGE_KEY = 'webSearchEnabled';
const LANGUAGE_STORAGE_KEY = 'generationLanguage';
const RECENT_OPEN_STORAGE_KEY = 'recentClassroomsOpen';

interface FormState {
  pdfFile: File | null;
  requirement: string;
  language: 'zh-CN' | 'en-US';
  webSearch: boolean;
}

const initialFormState: FormState = {
  pdfFile: null,
  requirement: '',
  language: 'zh-CN',
  webSearch: false,
};

// 这是一个叫 HomePage 的函数组件。
// 你可以把它理解成：
// “这个函数负责管理首页的数据，并且返回首页长什么样”
function HomePage() {
  // structureGuideEnabled 勾选标志位 setStructureGuideEnabled 设置勾选位
  const [structureGuideEnabled, setStructureGuideEnabled] = useState(false);
  // useI18n() 是“多语言工具”
  // 它返回一个对象，我们从里面拿出 3 个东西：
  // t         = 翻译函数，比如 t('home.slogan') 会拿到某条文字
  // locale    = 当前语言，比如 'zh-CN' 或 'en-US'
  // setLocale = 切换语言的函数
  const { t, locale, setLocale } = useI18n();

  // useTheme() 是“主题工具”
  // theme    = 当前主题，比如 light / dark / system
  // setTheme = 修改主题的函数
  const { theme, setTheme } = useTheme();
  // useRouter() 是页面跳转工具
  // 比如后面 router.push('/generation-preview') 就是跳到新页面
  const router = useRouter();

  // useState(...) 是 React 里“保存状态”的方法
  // form = 当前表单数据
  // setForm = 修改表单数据的方法
  // initialFormState = 初始默认值
  const [form, setForm] = useState<FormState>(initialFormState);

  // settingsOpen = 设置弹窗现在是不是打开
  // setSettingsOpen = 改变设置弹窗开关的方法
  const [settingsOpen, setSettingsOpen] = useState(false);

  // settingsSection = 设置弹窗默认打开哪一栏
  // 比如可能是“模型设置”、“PDF 设置”等
  // undefined 表示“暂时没有指定哪一栏”
  const [settingsSection, setSettingsSection] = useState<
    import('@/lib/types/settings').SettingsSection | undefined
  >(undefined);

  // Draft cache for requirement text
  // useDraftCache 是“草稿缓存工具”
  // cachedValue 被改名成 cachedRequirement，表示“之前缓存的需求文本”
  // updateCache 被改名成 updateRequirementCache，表示“更新需求草稿缓存”
  const { cachedValue: cachedRequirement, updateCache: updateRequirementCache } =
    useDraftCache<string>({ key: 'requirementDraft' });

  // currentModelId = 当前在设置里选中的模型 ID
  // 这里从全局 store 里读取 modelId
  const currentModelId = useSettingsStore((s) => s.modelId);

  // recentOpen = “最近课堂”区域是否展开
  // 默认值是 true，也就是默认展开
  const [recentOpen, setRecentOpen] = useState(true);

  // 这是一个 useEffect
  // 你可以把它理解成：“页面第一次加载完成以后，执行这里面的代码”
  // 后面的 [] 表示：只执行一次
  useEffect(() => {
    try {
      // 从浏览器本地存储 localStorage 里读取“最近课堂是否展开”
      const saved = localStorage.getItem(RECENT_OPEN_STORAGE_KEY);

      // 如果以前存过值，就恢复这个状态
      if (saved !== null) setRecentOpen(saved !== 'false');
    } catch {
      // 如果 localStorage 不可用，就忽略，不报错
    }

    try {
      // 读取 webSearch 的保存值
      const savedWebSearch = localStorage.getItem(WEB_SEARCH_STORAGE_KEY);

      // 读取 language 的保存值
      const savedLanguage = localStorage.getItem(LANGUAGE_STORAGE_KEY);

      // updates 是一个“临时修改对象”
      // 等会把要更新的字段先收集起来，最后一起 setForm
      const updates: Partial<FormState> = {};

      // 如果以前保存过 webSearch = true，就恢复为 true
      if (savedWebSearch === 'true') updates.webSearch = true;

      // 如果以前保存过合法语言，就恢复这个语言
      if (savedLanguage === 'zh-CN' || savedLanguage === 'en-US') {
        updates.language = savedLanguage;
      } else {
        // 否则用浏览器语言自动判断
        // 如果浏览器语言以 zh 开头，就用中文
        // 否则用英文
        const detected = navigator.language?.startsWith('zh') ? 'zh-CN' : 'en-US';
        updates.language = detected;
      }

      // 如果 updates 里真的有要改的东西
      if (Object.keys(updates).length > 0) {
        // 就把这些字段并到 form 里
        // prev 代表旧的 form
        // ...prev 表示“先复制旧的”
        // ...updates 表示“再用新的值覆盖”
        setForm((prev) => ({ ...prev, ...updates }));
      }
    } catch {
      // 如果 localStorage 不可用，就忽略
    }
  }, []);

  // 这里保存“上一次看到的缓存需求文本”
  const [prevCachedRequirement, setPrevCachedRequirement] = useState(cachedRequirement);

  // 如果“当前缓存文本”和“上一次记录的缓存文本”不一样
  if (cachedRequirement !== prevCachedRequirement) {
    // 先把“上一次记录”更新成新的缓存值
    setPrevCachedRequirement(cachedRequirement);

    // 如果新的缓存值不为空
    if (cachedRequirement) {
      // 就把它恢复到 form.requirement 里
      setForm((prev) => ({ ...prev, requirement: cachedRequirement }));
    }
  }

  // languageOpen = 语言下拉菜单是否展开
  const [languageOpen, setLanguageOpen] = useState(false);

  // themeOpen = 主题下拉菜单是否展开
  const [themeOpen, setThemeOpen] = useState(false);

  // error = 错误信息
  // null 表示当前没有错误
  const [error, setError] = useState<string | null>(null);

  // classrooms = 最近课堂列表
  const [classrooms, setClassrooms] = useState<StageListItem[]>([]);

  // thumbnails = 每个课堂的缩略图
  // Record<string, Slide> 可以理解成：
  // “一个字典，key 是字符串 id，value 是 Slide”
  const [thumbnails, setThumbnails] = useState<Record<string, Slide>>({});

  // pendingDeleteId = 当前正在等待确认删除的课堂 id
  // null 表示没有正在删除的课堂
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);

  // toolbarRef = 指向右上角工具栏 DOM 的引用
  const toolbarRef = useRef<HTMLDivElement>(null);

  // textareaRef = 指向文本输入框 DOM 的引用
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 这个 useEffect 的作用是：
  // 当语言菜单或主题菜单打开后，如果你点击它们外面，就自动关闭
  useEffect(() => {
    // 如果两个菜单都没打开，就不用监听点击外部，直接返回
    if (!languageOpen && !themeOpen) return;

    // handleClickOutside = 处理“点击外部”的函数
    const handleClickOutside = (e: MouseEvent) => {
      // 如果 toolbarRef.current 存在
      // 并且点击的目标不在工具栏里面
      if (toolbarRef.current && !toolbarRef.current.contains(e.target as Node)) {
        // 就把两个菜单都关掉
        setLanguageOpen(false);
        setThemeOpen(false);
      }
    };

    // 给整个 document 注册鼠标按下事件
    document.addEventListener('mousedown', handleClickOutside);

    // 返回的这个函数是“清理函数”
    // 当组件卸载或者依赖变化时，把监听器移除掉
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [languageOpen, themeOpen]);

  // loadClassrooms = 读取最近课堂列表的异步函数
  const loadClassrooms = async () => {
    try {
      // listStages() 会去拿课堂列表
      const list = await listStages();

      // 保存到 classrooms 状态里
      setClassrooms(list);

      // 如果列表不是空的
      if (list.length > 0) {
        // 先把所有 classroom 的 id 取出来
        // list.map((c) => c.id) 的意思就是：
        // 遍历 list，把每个 c 的 id 组成一个新数组
        const slides = await getFirstSlideByStages(list.map((c) => c.id));

        // 保存缩略图
        setThumbnails(slides);
      }
    } catch (err) {
      // 出错就记日志
      log.error('Failed to load classrooms:', err);
    }
  };

  // 这个 useEffect 在页面第一次加载时执行一次
  useEffect(() => {
    // 清理旧的媒体生成任务，避免不同课程之间的图片冲突
    useMediaGenerationStore.getState().revokeObjectUrls();
    useMediaGenerationStore.setState({ tasks: {} });

    // 再加载最近课堂列表
    loadClassrooms();
  }, []);

  // handleDelete = 点击删除按钮时调用
  // id = 要删的课堂 id
  // e  = 鼠标事件对象
  const handleDelete = (id: string, e: React.MouseEvent) => {
    // 阻止事件继续冒泡
    // 否则点删除时，可能会顺便触发整个卡片的点击跳转
    e.stopPropagation();

    // 把这个 id 设为“待确认删除”
    setPendingDeleteId(id);
  };

  // confirmDelete = 真正确认删除
  // id = 要删除的课堂 id
  const confirmDelete = async (id: string) => {
    // 先取消“待确认状态”
    setPendingDeleteId(null);

    try {
      // 删除数据
      await deleteStageData(id);

      // 删除完重新加载列表
      await loadClassrooms();
    } catch (err) {
      // 出错就记日志并弹提示
      log.error('Failed to delete classroom:', err);
      toast.error('Failed to delete classroom');
    }
  };

  // updateForm = 统一更新表单的函数
  // field = 要改哪个字段，比如 'language'、'requirement'
  // value = 这个字段的新值
  const updateForm = <K extends keyof FormState>(field: K, value: FormState[K]) => {
    // 更新 form
    setForm((prev) => ({ ...prev, [field]: value }));

    try {
      // 如果改的是 webSearch，就同步写入 localStorage
      if (field === 'webSearch') localStorage.setItem(WEB_SEARCH_STORAGE_KEY, String(value));

      // 如果改的是 language，就同步写入 localStorage
      if (field === 'language') localStorage.setItem(LANGUAGE_STORAGE_KEY, String(value));

      // 如果改的是 requirement，就更新草稿缓存
      if (field === 'requirement') updateRequirementCache(value as string);
    } catch {
      // 忽略本地存储失败
    }
  };

  // showSetupToast = 弹一个“请先配置模型”的提示框
  // icon  = 提示框左边显示的图标
  // title = 提示框主标题
  // desc  = 提示框说明文字
  const showSetupToast = (icon: React.ReactNode, title: string, desc: string) => {
    toast.custom(
      (id) => (
        <div
          // className 是样式，不是核心逻辑
          className="w-[356px] rounded-xl border border-amber-200/60 dark:border-amber-800/40 bg-gradient-to-r from-amber-50 via-white to-amber-50 dark:from-amber-950/60 dark:via-slate-900 dark:to-amber-950/60 shadow-lg shadow-amber-500/8 dark:shadow-amber-900/20 p-4 flex items-start gap-3 cursor-pointer"
          onClick={() => {
            // 点击这个 toast 时，先关闭 toast
            toast.dismiss(id);

            // 再打开设置弹窗
            setSettingsOpen(true);
          }}
        >
          <div className="shrink-0 mt-0.5 size-9 rounded-lg bg-amber-100 dark:bg-amber-900/40 flex items-center justify-center ring-1 ring-amber-200/50 dark:ring-amber-800/30">
            {icon}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-amber-900 dark:text-amber-200 leading-tight">
              {title}
            </p>
            <p className="text-xs text-amber-700/80 dark:text-amber-400/70 mt-0.5 leading-relaxed">
              {desc}
            </p>
          </div>
          <div className="shrink-0 mt-1 text-[10px] font-medium text-amber-500 dark:text-amber-500/70 tracking-wide">
            <Settings className="size-3.5 animate-[spin_3s_linear_infinite]" />
          </div>
        </div>
      ),
      {
        // 4000 表示这个 toast 显示 4000 毫秒，也就是 4 秒
        duration: 4000,
      },
    );
  };

  // handleGenerate = 点击“进入课堂”时的核心函数
  const handleGenerate = async () => {
    // 先检查模型是不是已经配置了
    if (!currentModelId) {
      // 如果没有模型，就弹出提示
      showSetupToast(
        // 第 1 个参数：图标
        <BotOff className="size-4.5 text-amber-600 dark:text-amber-400" />,

        // 第 2 个参数：标题
        t('settings.modelNotConfigured'),

        // 第 3 个参数：说明
        t('settings.setupNeeded'),
      );

      // 同时打开设置页
      setSettingsOpen(true);

      // 然后直接结束，不继续生成
      return;
    }

    // 如果 requirement 去掉前后空格后还是空
    if (!form.requirement.trim()) {
      // 就设置错误信息
      setError(t('upload.requirementRequired'));
      return;
    }

    // 清掉旧错误
    setError(null);

    try {
      // 取当前用户资料
      const userProfile = useUserProfileStore.getState();

      // requirements = 后续生成课时真正要用到的输入对象
      const requirements: UserRequirements = {
        // requirement = 用户输入的需求文本
        requirement: form.requirement,

        // language = 生成时用的语言
        language: form.language,

        // userNickname = 用户昵称，如果没有就用 undefined
        userNickname: userProfile.nickname || undefined,

        // userBio = 用户简介，如果没有就用 undefined
        userBio: userProfile.bio || undefined,

        // webSearch = 是否允许联网搜索，如果 false 就传 undefined
        webSearch: form.webSearch || undefined,
      };

      // 下面这些变量先定义出来，后面如果有 PDF 再填
      let pdfStorageKey: string | undefined;
      let pdfFileName: string | undefined;
      let pdfProviderId: string | undefined;
      let pdfProviderConfig: { apiKey?: string; baseUrl?: string } | undefined;

      // 如果用户上传了 PDF
      if (form.pdfFile) {
        // 存储 PDF，并得到 storage key
        pdfStorageKey = await storePdfBlob(form.pdfFile);

        // 保存文件名
        pdfFileName = form.pdfFile.name;

        // 取设置中的 PDF provider
        const settings = useSettingsStore.getState();
        pdfProviderId = settings.pdfProviderId;

        // 读取这个 provider 的配置
        const providerCfg = settings.pdfProvidersConfig?.[settings.pdfProviderId];

        // 如果配置存在
        if (providerCfg) {
          pdfProviderConfig = {
            // API key
            apiKey: providerCfg.apiKey,

            // base URL
            baseUrl: providerCfg.baseUrl,
          };
        }
      }

      // sessionState = 本次生成流程的会话状态
      const sessionState = {
        // sessionId = 一个随机生成的唯一 id
        sessionId: nanoid(),

        // requirements = 刚刚整理好的用户需求
        requirements,

        // pdfText = 先留空，后续可能解析 PDF 文本
        pdfText: '',

        // pdfImages = 先留空，后续可能解析 PDF 图片
        pdfImages: [],

        // imageStorageIds = 先留空，后续可能保存图片 id
        imageStorageIds: [],

        // 如果有 PDF，这里会带上 PDF 的存储 key
        pdfStorageKey,

        // PDF 文件名
        pdfFileName,

        // PDF provider id
        pdfProviderId,

        // PDF provider 配置
        pdfProviderConfig,

        // sceneOutlines = 课程场景提纲，初始是 null
        sceneOutlines: null,

        // currentStep = 当前步骤，固定写成 'generating'
        currentStep: 'generating' as const,
        // 增加一个字段用于接受课程架构 ycf
        courseBlueprint: null,

        // 新增：是否启用真实课程架构 ycf
        structureGuideEnabled,
      };

      // 把 sessionState 存进 sessionStorage
      // JSON.stringify(...) 的作用是把对象转成字符串
      sessionStorage.setItem('generationSession', JSON.stringify(sessionState));

      // 跳转到生成预览页
      router.push('/generation-preview');
    } catch (err) {
      // 如果上面流程出错，就记录日志
      log.error('Error preparing generation:', err);

      // 再给用户设置错误信息
      setError(err instanceof Error ? err.message : t('upload.generateFailed'));
    }
  };

  // formatDate = 把时间戳变成更友好的日期文本
  // timestamp = 一个数字形式的时间
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();

    // diffTime = 当前时间和目标时间的差值（毫秒）
    const diffTime = Math.abs(now.getTime() - date.getTime());

    // diffDays = 相差多少天
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    // 如果是今天
    if (diffDays === 0) return t('classroom.today');

    // 如果是昨天
    if (diffDays === 1) return t('classroom.yesterday');

    // 如果小于 7 天
    if (diffDays < 7) return `${diffDays} ${t('classroom.daysAgo')}`;

    // 否则就直接返回本地日期格式
    return date.toLocaleDateString();
  };

  // canGenerate = 是否允许点击“进入课堂”
  // !! 的意思是把结果强制变成 true / false
  // 只要 requirement 不为空，就可以生成
  const canGenerate = !!form.requirement.trim();

  // handleKeyDown = 文本框键盘按下时触发
  // e = 键盘事件对象
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // 如果按的是 Command+Enter（Mac）或者 Ctrl+Enter（Windows）
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      // 阻止默认行为
      e.preventDefault();

      // 如果当前允许生成
      if (canGenerate) handleGenerate();
    }
  };

  // return (...) 表示“这个组件最终要渲染成什么样”
  return (
    <div className="min-h-[100dvh] w-full bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex flex-col items-center p-4 pt-16 md:p-8 md:pt-16 overflow-x-hidden">
      {/* 最外层大容器：只是整体页面背景和布局 */}

      {/* 右上角工具条 */}
      <div
        // ref={toolbarRef} 表示：把这个 DOM 记到 toolbarRef 里
        ref={toolbarRef}

        // className 只是样式
        className="fixed top-4 right-4 z-50 flex items-center gap-1 bg-white/60 dark:bg-gray-800/60 backdrop-blur-md px-2 py-1.5 rounded-full border border-gray-100/50 dark:border-gray-700/50 shadow-sm"
      >
        {/* Language Selector = 语言选择器 */}
        <div className="relative">
          <button
            onClick={() => {
              // 点击时切换语言菜单开关
              setLanguageOpen(!languageOpen);

              // 同时关闭主题菜单
              setThemeOpen(false);
            }}
            className="flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-bold text-gray-500 dark:text-gray-400 hover:bg-white dark:hover:bg-gray-700 hover:text-gray-800 dark:hover:text-gray-200 hover:shadow-sm transition-all"
          >
            {/* 如果当前语言是中文，就显示 CN，否则显示 EN */}
            {locale === 'zh-CN' ? 'CN' : 'EN'}
          </button>

          {/* 如果 languageOpen 为 true，就显示下拉框 */}
          {languageOpen && (
            <div className="absolute top-full mt-2 right-0 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden z-50 min-w-[120px]">
              <button
                onClick={() => {
                  // 选中文
                  setLocale('zh-CN');

                  // 关掉下拉菜单
                  setLanguageOpen(false);
                }}
                className={cn(
                  'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
                  locale === 'zh-CN' &&
                    'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                )}
              >
                简体中文
              </button>

              <button
                onClick={() => {
                  // 选英文
                  setLocale('en-US');

                  // 关掉下拉菜单
                  setLanguageOpen(false);
                }}
                className={cn(
                  'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
                  locale === 'en-US' &&
                    'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                )}
              >
                English
              </button>
            </div>
          )}
        </div>

        {/* 这一条竖线只是分隔线 */}
        <div className="w-[1px] h-4 bg-gray-200 dark:bg-gray-700" />

        {/* Theme Selector = 主题选择器 */}
        <div className="relative">
          <button
            onClick={() => {
              // 点击时切换主题菜单开关
              setThemeOpen(!themeOpen);

              // 同时关闭语言菜单
              setLanguageOpen(false);
            }}
            className="p-2 rounded-full text-gray-400 dark:text-gray-500 hover:bg-white dark:hover:bg-gray-700 hover:text-gray-800 dark:hover:text-gray-200 hover:shadow-sm transition-all"
          >
            {/* 根据当前 theme 显示不同图标 */}
            {theme === 'light' && <Sun className="w-4 h-4" />}
            {theme === 'dark' && <Moon className="w-4 h-4" />}
            {theme === 'system' && <Monitor className="w-4 h-4" />}
          </button>

          {/* 如果 themeOpen 为 true，就显示主题下拉菜单 */}
          {themeOpen && (
            <div className="absolute top-full mt-2 right-0 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden z-50 min-w-[140px]">
              <button
                onClick={() => {
                  setTheme('light');
                  setThemeOpen(false);
                }}
                className={cn(
                  'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2',
                  theme === 'light' &&
                    'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                )}
              >
                <Sun className="w-4 h-4" />
                {t('settings.themeOptions.light')}
              </button>

              <button
                onClick={() => {
                  setTheme('dark');
                  setThemeOpen(false);
                }}
                className={cn(
                  'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2',
                  theme === 'dark' &&
                    'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                )}
              >
                <Moon className="w-4 h-4" />
                {t('settings.themeOptions.dark')}
              </button>

              <button
                onClick={() => {
                  setTheme('system');
                  setThemeOpen(false);
                }}
                className={cn(
                  'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2',
                  theme === 'system' &&
                    'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                )}
              >
                <Monitor className="w-4 h-4" />
                {t('settings.themeOptions.system')}
              </button>
            </div>
          )}
        </div>

        {/* 分隔线 */}
        <div className="w-[1px] h-4 bg-gray-200 dark:bg-gray-700" />

        {/* Settings Button = 设置按钮 */}
        <div className="relative">
          <button
            onClick={() => setSettingsOpen(true)}
            className="p-2 rounded-full text-gray-400 dark:text-gray-500 hover:bg-white dark:hover:bg-gray-700 hover:text-gray-800 dark:hover:text-gray-200 hover:shadow-sm transition-all group"
          >
            <Settings className="w-4 h-4 group-hover:rotate-90 transition-transform duration-500" />
          </button>
        </div>
      </div>

      {/* 这是设置弹窗组件 */}
      <SettingsDialog
        // open = 这个弹窗是否打开
        open={settingsOpen}

        // onOpenChange = 当弹窗开关变化时调用
        onOpenChange={(open) => {
          // 同步 settingsOpen
          setSettingsOpen(open);

          // 如果弹窗关闭了，就把默认 section 清空
          if (!open) setSettingsSection(undefined);
        }}

        // initialSection = 一开始默认打开哪一栏
        initialSection={settingsSection}
      />

      {/* 背景装饰层，只是视觉效果 */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDuration: '4s' }}
        />
        <div
          className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDuration: '6s' }}
        />
      </div>

      {/* Hero 主区域：logo + 输入框 + 工具区 */}
      <motion.div
        // 下面三个参数是入场动画
        initial={{ opacity: 0, y: 20 }}   // 初始：透明、往下偏 20
        animate={{ opacity: 1, y: 0 }}    // 动画结束：不透明、回到原位
        transition={{ duration: 0.6, ease: 'easeOut' }} // 动画时长和曲线

        className={cn(
          'relative z-20 w-full max-w-[800px] flex flex-col items-center',

          // 如果没有 classrooms，就让这个区域垂直居中
          classrooms.length === 0 ? 'justify-center min-h-[calc(100dvh-8rem)]' : 'mt-[10vh]',
        )}
      >
        {/* Logo */}
        <motion.img
          src="/logo-horizontal.png"
          alt="OpenMAIC"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            delay: 0.1,
            type: 'spring',
            stiffness: 200,
            damping: 20,
          }}
          className="h-12 md:h-16 mb-2 -ml-2 md:-ml-3"
        />

        {/* 标语 */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.25 }}
          className="text-sm text-muted-foreground/60 mb-8"
        >
          {t('home.slogan')}
        </motion.p>

        {/* 整个输入卡片 */}
        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.35 }}
          className="w-full"
        >
          <div className="w-full rounded-2xl border border-border/60 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl shadow-xl shadow-black/[0.03] dark:shadow-black/20 transition-shadow focus-within:shadow-2xl focus-within:shadow-violet-500/[0.06]">
            {/* 上方：问候栏 + agent 栏 */}
            <div className="relative z-20 flex items-start justify-between">
              <GreetingBar />
              <div className="pr-3 pt-3.5 shrink-0">
                <AgentBar />
              </div>
            </div>

            {/* 文本输入框 */}
            <textarea
              // ref = 保存这个 textarea 的 DOM 引用
              ref={textareaRef}

              // placeholder = 占位提示文字
              placeholder={t('upload.requirementPlaceholder')}

              // className = 样式
              className="w-full resize-none border-0 bg-transparent px-4 pt-1 pb-2 text-[13px] leading-relaxed placeholder:text-muted-foreground/40 focus:outline-none min-h-[140px] max-h-[300px]"

              // value = 文本框当前显示的值
              value={form.requirement}

              // onChange = 用户输入时触发
              // e.target.value 就是输入框的新内容
              onChange={(e) => updateForm('requirement', e.target.value)}

              // onKeyDown = 按键时触发
              onKeyDown={handleKeyDown}

              // rows = 默认显示几行高
              rows={4}
            />

            {/*<div> bottom 标签*/}
            {/*p 内边距 x 左右 b 下 flex 弹性布局 items-end底部对齐*/}
            {/* Toolbar row = 工具条这一行 */}
            {/*flex-1 尽量占满空间   min-w-0 允许最小化<div className="flex-1 min-w-0">*/}
            {/*最下面一行的类，调这里可以改下面的按钮*/}
            <div className="px-3 pb-3 flex items-end gap-2">
            <div className="flex-1 min-w-0">
                <GenerationToolbar
                  // language = 当前语言
                  language={form.language}

                  // onLanguageChange = 工具栏里改语言时怎么处理
                  onLanguageChange={(lang) => updateForm('language', lang)}

                  // webSearch = 当前是否启用联网搜索
                  webSearch={form.webSearch}

                  // onWebSearchChange = 工具栏里切换联网搜索时怎么处理
                  onWebSearchChange={(v) => updateForm('webSearch', v)}

                  // onSettingsOpen = 当工具栏里想打开设置时
                  onSettingsOpen={(section) => {
                    setSettingsSection(section);
                    setSettingsOpen(true);
                  }}

                  // pdfFile = 当前上传的 PDF
                  pdfFile={form.pdfFile}

                  // onPdfFileChange = PDF 文件变化时怎么更新 form
                  onPdfFileChange={(f) => updateForm('pdfFile', f)}

                  // onPdfError = PDF 出错时怎么设置错误信息
                  onPdfError={setError}
                />
              </div>

              {/* 语音输入按钮 */}
              <SpeechButton
                size="md"
                onTranscription={(text) => {
                  // 当语音转文字成功后，把 text 拼到 requirement 后面
                  setForm((prev) => {
                    const next = prev.requirement + (prev.requirement ? ' ' : '') + text;

                    // 同时更新草稿缓存
                    updateRequirementCache(next);

                    // 返回新的 form
                    return { ...prev, requirement: next };
                  });
                }}
              />
              {/*课程结构按钮，用lable是因为有字也有图像*/}
              {/* onChange 复选框状态改变后执行这里的代码*/}
              <label className="shrink-0 h-8 rounded-lg border border-violet-300 bg-violet-50 px-3 text-xs font-medium text-violet-700 hover:bg-violet-100 transition-colors flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={structureGuideEnabled}

                  onChange={(e) => {
                    const checked = e.target.checked;
                    setStructureGuideEnabled(checked);

                    if (checked) {
                      alert('已勾选：后续将启用真实课程架构');
                    }
                  }}
                />
                <span>真实课程架构</span>
              </label>

              {/* 发送按钮 */}
              <button
                // 点击时执行 handleGenerate()
                onClick={handleGenerate}

                // disabled = true 时按钮不能点
                disabled={!canGenerate}

                className={cn(
                  'shrink-0 h-8 rounded-lg flex items-center justify-center gap-1.5 transition-all px-3',
                  canGenerate
                    ? 'bg-primary text-primary-foreground hover:opacity-90 shadow-sm cursor-pointer'
                    : 'bg-muted text-muted-foreground/40 cursor-not-allowed',
                )}
              >
                <span className="text-xs font-medium">{t('toolbar.enterClassroom')}</span>
                <ArrowUp className="size-3.5" />
              </button>
            </div>
          </div>
        </motion.div>

        {/* 错误信息区域 */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-3 w-full p-3 bg-destructive/10 border border-destructive/20 rounded-lg"
            >
              <p className="text-sm text-destructive">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* 最近课堂区域：只有 classrooms 不为空时才显示 */}
      {classrooms.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="relative z-10 mt-10 w-full max-w-6xl flex flex-col items-center"
        >
          {/* 展开 / 收起按钮 */}
          <button
            onClick={() => {
              const next = !recentOpen;
              setRecentOpen(next);

              try {
                localStorage.setItem(RECENT_OPEN_STORAGE_KEY, String(next));
              } catch {
                /* ignore */
              }
            }}
            className="group w-full flex items-center gap-4 py-2 cursor-pointer"
          >
            <div className="flex-1 h-px bg-border/40 group-hover:bg-border/70 transition-colors" />
            <span className="shrink-0 flex items-center gap-2 text-[13px] text-muted-foreground/60 group-hover:text-foreground/70 transition-colors select-none">
              <Clock className="size-3.5" />
              {t('classroom.recentClassrooms')}
              <span className="text-[11px] tabular-nums opacity-60">{classrooms.length}</span>
              <motion.div
                animate={{ rotate: recentOpen ? 180 : 0 }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
              >
                <ChevronDown className="size-3.5" />
              </motion.div>
            </span>
            <div className="flex-1 h-px bg-border/40 group-hover:bg-border/70 transition-colors" />
          </button>

          {/* 最近课堂展开内容 */}
          <AnimatePresence>
            {recentOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
                className="w-full overflow-hidden"
              >
                <div className="pt-8 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-5 gap-y-8">
                  {classrooms.map((classroom, i) => (
                    <motion.div
                      key={classroom.id}
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{
                        delay: i * 0.04,
                        duration: 0.35,
                        ease: 'easeOut',
                      }}
                    >
                      <ClassroomCard
                        // classroom = 当前课堂数据
                        classroom={classroom}

                        // slide = 当前课堂对应的缩略图
                        slide={thumbnails[classroom.id]}

                        // formatDate = 格式化日期的函数
                        formatDate={formatDate}

                        // onDelete = 点击删除按钮时调用
                        onDelete={handleDelete}

                        // confirmingDelete = 这个课堂是不是正在等待删除确认
                        confirmingDelete={pendingDeleteId === classroom.id}

                        // onConfirmDelete = 确认删除时调用
                        onConfirmDelete={() => confirmDelete(classroom.id)}

                        // onCancelDelete = 取消删除时调用
                        onCancelDelete={() => setPendingDeleteId(null)}

                        // onClick = 点击卡片时，跳到对应课堂页面
                        onClick={() => router.push(`/classroom/${classroom.id}`)}
                      />
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* 页脚 */}
      <div className="mt-auto pt-12 pb-4 text-center text-xs text-muted-foreground/40">
        OpenMAIC Open Source Project
      </div>
    </div>
  );
}

// ─── Greeting Bar — avatar + "Hi, Name", click to edit in-place ────
const MAX_AVATAR_SIZE = 5 * 1024 * 1024;

function isCustomAvatar(src: string) {
  return src.startsWith('data:');
}

function GreetingBar() {
  const { t } = useI18n();
  const avatar = useUserProfileStore((s) => s.avatar);
  const nickname = useUserProfileStore((s) => s.nickname);
  const bio = useUserProfileStore((s) => s.bio);
  const setAvatar = useUserProfileStore((s) => s.setAvatar);
  const setNickname = useUserProfileStore((s) => s.setNickname);
  const setBio = useUserProfileStore((s) => s.setBio);

  const [open, setOpen] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState('');
  const [avatarPickerOpen, setAvatarPickerOpen] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const avatarInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const displayName = nickname || t('profile.defaultNickname');

  // Click-outside to collapse
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        setEditingName(false);
        setAvatarPickerOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const startEditName = () => {
    setNameDraft(nickname);
    setEditingName(true);
    setTimeout(() => nameInputRef.current?.focus(), 50);
  };

  const commitName = () => {
    setNickname(nameDraft.trim());
    setEditingName(false);
  };

  const handleAvatarUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_AVATAR_SIZE) {
      toast.error(t('profile.fileTooLarge'));
      return;
    }
    if (!file.type.startsWith('image/')) {
      toast.error(t('profile.invalidFileType'));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const img = new window.Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d')!;
        const scale = Math.max(128 / img.width, 128 / img.height);
        const w = img.width * scale;
        const h = img.height * scale;
        ctx.drawImage(img, (128 - w) / 2, (128 - h) / 2, w, h);
        setAvatar(canvas.toDataURL('image/jpeg', 0.85));
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  return (
    <div ref={containerRef} className="relative pl-4 pr-2 pt-3.5 pb-1 w-auto">
      <input
        ref={avatarInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleAvatarUpload}
      />

      {/* ── Collapsed pill (always in flow) ── */}
      {!open && (
        <div
          className="flex items-center gap-2.5 cursor-pointer transition-all duration-200 group rounded-full px-2.5 py-1.5 border border-border/50 text-muted-foreground/70 hover:text-foreground hover:bg-muted/60 active:scale-[0.97]"
          onClick={() => setOpen(true)}
        >
          <div className="shrink-0 relative">
            <div className="size-8 rounded-full overflow-hidden ring-[1.5px] ring-border/30 group-hover:ring-violet-400/60 dark:group-hover:ring-violet-400/40 transition-all duration-300">
              <img src={avatar} alt="" className="size-full object-cover" />
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 size-3.5 rounded-full bg-white dark:bg-slate-800 border border-border/40 flex items-center justify-center opacity-60 group-hover:opacity-100 transition-opacity">
              <Pencil className="size-[7px] text-muted-foreground/70" />
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="leading-none select-none flex items-center gap-1">
                  <span>
                    <span className="text-xs text-muted-foreground/60 group-hover:text-muted-foreground transition-colors">
                      {t('home.greeting')}
                    </span>
                    <span className="text-[13px] font-semibold text-foreground/85 group-hover:text-foreground transition-colors">
                      {displayName}
                    </span>
                  </span>
                  <ChevronDown className="size-3 text-muted-foreground/30 group-hover:text-muted-foreground/60 transition-colors shrink-0" />
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={4}>
                {t('profile.editTooltip')}
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
      )}

      {/* ── Expanded panel (absolute, floating) ── */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.97 }}
            transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
            className="absolute left-4 top-3.5 z-50 w-64"
          >
            <div className="rounded-2xl bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm ring-1 ring-black/[0.04] dark:ring-white/[0.06] shadow-[0_1px_8px_-2px_rgba(0,0,0,0.06)] dark:shadow-[0_1px_8px_-2px_rgba(0,0,0,0.3)] px-2.5 py-2">
              {/* ── Row: avatar + name ── */}
              <div
                className="flex items-center gap-2.5 cursor-pointer transition-all duration-200"
                onClick={() => {
                  setOpen(false);
                  setEditingName(false);
                  setAvatarPickerOpen(false);
                }}
              >
                {/* Avatar */}
                <div
                  className="shrink-0 relative cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    setAvatarPickerOpen(!avatarPickerOpen);
                  }}
                >
                  <div className="size-8 rounded-full overflow-hidden ring-[1.5px] ring-violet-300/70 dark:ring-violet-500/40 transition-all duration-300">
                    <img src={avatar} alt="" className="size-full object-cover" />
                  </div>
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute -bottom-0.5 -right-0.5 size-3.5 rounded-full bg-white dark:bg-slate-800 border border-border/60 flex items-center justify-center"
                  >
                    <ChevronDown
                      className={cn(
                        'size-2 text-muted-foreground/70 transition-transform duration-200',
                        avatarPickerOpen && 'rotate-180',
                      )}
                    />
                  </motion.div>
                </div>

                {/* Text */}
                <div className="flex-1 min-w-0">
                  {editingName ? (
                    <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                      <input
                        ref={nameInputRef}
                        value={nameDraft}
                        onChange={(e) => setNameDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') commitName();
                          if (e.key === 'Escape') {
                            setEditingName(false);
                          }
                        }}
                        onBlur={commitName}
                        maxLength={20}
                        placeholder={t('profile.defaultNickname')}
                        className="flex-1 min-w-0 h-6 bg-transparent border-b border-border/80 text-[13px] font-semibold text-foreground outline-none placeholder:text-muted-foreground/40"
                      />
                      <button
                        onClick={commitName}
                        className="shrink-0 size-5 rounded flex items-center justify-center text-violet-500 hover:bg-violet-100 dark:hover:bg-violet-900/30"
                      >
                        <Check className="size-3" />
                      </button>
                    </div>
                  ) : (
                    <span
                      onClick={(e) => {
                        e.stopPropagation();
                        startEditName();
                      }}
                      className="group/name inline-flex items-center gap-1 cursor-pointer"
                    >
                      <span className="text-[13px] font-semibold text-foreground/85 group-hover/name:text-foreground transition-colors">
                        {displayName}
                      </span>
                      <Pencil className="size-2.5 text-muted-foreground/30 opacity-0 group-hover/name:opacity-100 transition-opacity" />
                    </span>
                  )}
                </div>

                {/* Collapse arrow */}
                <motion.div
                  initial={{ opacity: 0, y: -2 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="shrink-0 size-6 rounded-full flex items-center justify-center hover:bg-black/[0.04] dark:hover:bg-white/[0.06] transition-colors"
                >
                  <ChevronUp className="size-3.5 text-muted-foreground/50" />
                </motion.div>
              </div>

              {/* ── Expandable content ── */}
              <div className="pt-2" onClick={(e) => e.stopPropagation()}>
                {/* Avatar picker */}
                <AnimatePresence>
                  {avatarPickerOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.15, ease: 'easeInOut' }}
                      className="overflow-hidden"
                    >
                      <div className="p-1 pb-2.5 flex items-center gap-1.5 flex-wrap">
                        {AVATAR_OPTIONS.map((url) => (
                          <button
                            key={url}
                            onClick={() => setAvatar(url)}
                            className={cn(
                              'size-7 rounded-full overflow-hidden bg-gray-50 dark:bg-gray-800 cursor-pointer transition-all duration-150',
                              'hover:scale-110 active:scale-95',
                              avatar === url
                                ? 'ring-2 ring-violet-400 dark:ring-violet-500 ring-offset-0'
                                : 'hover:ring-1 hover:ring-muted-foreground/30',
                            )}
                          >
                            <img src={url} alt="" className="size-full" />
                          </button>
                        ))}
                        <label
                          className={cn(
                            'size-7 rounded-full flex items-center justify-center cursor-pointer transition-all duration-150 border border-dashed',
                            'hover:scale-110 active:scale-95',
                            isCustomAvatar(avatar)
                              ? 'ring-2 ring-violet-400 dark:ring-violet-500 ring-offset-0 border-violet-300 dark:border-violet-600 bg-violet-50 dark:bg-violet-900/30'
                              : 'border-muted-foreground/30 text-muted-foreground/50 hover:border-muted-foreground/50',
                          )}
                          onClick={() => avatarInputRef.current?.click()}
                          title={t('profile.uploadAvatar')}
                        >
                          <ImagePlus className="size-3" />
                        </label>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Bio */}
                <UITextarea
                  value={bio}
                  onChange={(e) => setBio(e.target.value)}
                  placeholder={t('profile.bioPlaceholder')}
                  maxLength={200}
                  rows={2}
                  className="resize-none border-border/40 bg-transparent min-h-[72px] !text-[13px] !leading-relaxed placeholder:!text-[11px] placeholder:!leading-relaxed focus-visible:ring-1 focus-visible:ring-border/60"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Classroom Card — clean, minimal style ──────────────────────
function ClassroomCard({
  classroom,
  slide,
  formatDate,
  onDelete,
  confirmingDelete,
  onConfirmDelete,
  onCancelDelete,
  onClick,
}: {
  classroom: StageListItem;
  slide?: Slide;
  formatDate: (ts: number) => string;
  onDelete: (id: string, e: React.MouseEvent) => void;
  confirmingDelete: boolean;
  onConfirmDelete: () => void;
  onCancelDelete: () => void;
  onClick: () => void;
}) {
  const { t } = useI18n();
  const thumbRef = useRef<HTMLDivElement>(null);
  const [thumbWidth, setThumbWidth] = useState(0);

  useEffect(() => {
    const el = thumbRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setThumbWidth(Math.round(entry.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <div className="group cursor-pointer" onClick={confirmingDelete ? undefined : onClick}>
      {/* Thumbnail — large radius, no border, subtle bg */}
      <div
        ref={thumbRef}
        className="relative w-full aspect-[16/9] rounded-2xl bg-slate-100 dark:bg-slate-800/80 overflow-hidden transition-transform duration-200 group-hover:scale-[1.02]"
      >
        {slide && thumbWidth > 0 ? (
          <ThumbnailSlide
            slide={slide}
            size={thumbWidth}
            viewportSize={slide.viewportSize ?? 1000}
            viewportRatio={slide.viewportRatio ?? 0.5625}
          />
        ) : !slide ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="size-12 rounded-2xl bg-gradient-to-br from-violet-100 to-blue-100 dark:from-violet-900/30 dark:to-blue-900/30 flex items-center justify-center">
              <span className="text-xl opacity-50">📄</span>
            </div>
          </div>
        ) : null}

        {/* Delete — top-right, only on hover */}
        <AnimatePresence>
          {!confirmingDelete && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <Button
                size="icon"
                variant="ghost"
                className="absolute top-2 right-2 size-7 opacity-0 group-hover:opacity-100 transition-opacity bg-black/30 hover:bg-destructive/80 text-white hover:text-white backdrop-blur-sm rounded-full"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(classroom.id, e);
                }}
              >
                <Trash2 className="size-3.5" />
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Inline delete confirmation overlay */}
        <AnimatePresence>
          {confirmingDelete && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-black/50 backdrop-blur-[6px]"
              onClick={(e) => e.stopPropagation()}
            >
              <span className="text-[13px] font-medium text-white/90">
                {t('classroom.deleteConfirmTitle')}?
              </span>
              <div className="flex gap-2">
                <button
                  className="px-3.5 py-1 rounded-lg text-[12px] font-medium bg-white/15 text-white/80 hover:bg-white/25 backdrop-blur-sm transition-colors"
                  onClick={onCancelDelete}
                >
                  {t('common.cancel')}
                </button>
                <button
                  className="px-3.5 py-1 rounded-lg text-[12px] font-medium bg-red-500/90 text-white hover:bg-red-500 transition-colors"
                  onClick={onConfirmDelete}
                >
                  {t('classroom.delete')}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Info — outside the thumbnail */}
      <div className="mt-2.5 px-1 flex items-center gap-2">
        <span className="shrink-0 inline-flex items-center rounded-full bg-violet-100 dark:bg-violet-900/30 px-2 py-0.5 text-[11px] font-medium text-violet-600 dark:text-violet-400">
          {classroom.sceneCount} {t('classroom.slides')} · {formatDate(classroom.updatedAt)}
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <p className="font-medium text-[15px] truncate text-foreground/90 min-w-0">
              {classroom.name}
            </p>
          </TooltipTrigger>
          <TooltipContent
            side="bottom"
            sideOffset={4}
            className="!max-w-[min(90vw,32rem)] break-words whitespace-normal"
          >
            <div className="flex items-center gap-1.5">
              <span className="break-all">{classroom.name}</span>
              <button
                className="shrink-0 p-0.5 rounded hover:bg-foreground/10 transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  navigator.clipboard.writeText(classroom.name);
                  toast.success(t('classroom.nameCopied'));
                }}
              >
                <Copy className="size-3 opacity-60" />
              </button>
            </div>
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}

export default function Page() {
  return <HomePage />;
}
