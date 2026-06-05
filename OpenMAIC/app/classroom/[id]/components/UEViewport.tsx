'use client';

import type { TeacherStatus } from './TeacherPanel';

type UEViewportProps = {
  teacherStatus: TeacherStatus;
  streamUrl?: string;
  connected?: boolean;
  iframeRef?: React.RefObject<HTMLIFrameElement | null>;
  mocapInfo?: {
    mocapId?: string;
    mocapAction?: string;
    mappedAction?: TeacherStatus;
    ueAction?: string;
  } | null;
};

const statusLabelMap: Record<TeacherStatus, string> = {
  Idle: '待命中',
  Talking: '讲解中',
  Pointing: '指向重点',
  Thinking: '思考中',
  MC05: '侧身讲解',
  MC06: '侧身思考',
  MC07: '侧身指向',
  MC08: '高位指向',
  MC09: '侧身挥掌',
  MC10: 'PPT 操作',
  MC11: '转身过渡',
  MC12: '板书书写',
  MC13: '上下书写',
  MC14: '边写边讲',
  MC15: '遮挡板书',
};

export function UEViewport({
  teacherStatus,
  streamUrl,
  connected = false,
  iframeRef,
  mocapInfo,
}: UEViewportProps) {
  console.log('[UEViewport] render teacherStatus =', teacherStatus);

  const mappedAction = mocapInfo?.mappedAction || teacherStatus;
  const ueAction = mocapInfo?.ueAction || mappedAction;
  const mocapLabel =
    mocapInfo?.mocapId || mocapInfo?.mocapAction
      ? `${mocapInfo?.mocapId || '--'} ${mocapInfo?.mocapAction || ''}`.trim()
      : '暂无 MC 动作';

  return (
    <div className="w-full h-full min-h-0 rounded-2xl border border-slate-200/70 dark:border-slate-700/70 bg-white/80 dark:bg-slate-900/70 backdrop-blur p-4 shadow-sm flex flex-col">
      <div className="shrink-0 flex items-center justify-between">
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">UE 教师窗口</div>

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${
              connected ? 'bg-green-500' : 'bg-slate-400'
            }`}
          />
          {connected ? '已连接' : '未连接'}
        </div>
      </div>

      <div className="mt-4 flex-1 min-h-0 overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700 bg-black/90 flex items-center justify-center">
        {streamUrl ? (
          <iframe
            ref={iframeRef}
            src={streamUrl}
            title="UE Pixel Streaming"
            className="h-full w-full border-0"
            allow="autoplay; fullscreen; camera; microphone"
          />
        ) : (
          <div className="text-center px-4">
            <div className="text-sm font-medium text-white">等待 UE 画面接入</div>
            <div className="mt-2 text-xs text-slate-400 break-all">
              Pixel Streaming URL 尚未配置
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 shrink-0 grid grid-cols-3 gap-2 text-xs">
        <div className="min-w-0 rounded-lg bg-slate-100 dark:bg-slate-800 px-2.5 py-2">
          <div className="text-[10px] text-muted-foreground">MC 动作</div>
          <div
            className="mt-1 truncate font-medium text-slate-800 dark:text-slate-100"
            title={mocapLabel}
          >
            {mocapLabel}
          </div>
        </div>

        <div className="min-w-0 rounded-lg bg-blue-50 dark:bg-blue-950/40 px-2.5 py-2">
          <div className="text-[10px] text-blue-500 dark:text-blue-300">映射动作</div>
          <div className="mt-1 truncate font-medium text-blue-800 dark:text-blue-200">
            {mappedAction} · {statusLabelMap[mappedAction]}
          </div>
        </div>

        <div className="min-w-0 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 px-2.5 py-2">
          <div className="text-[10px] text-emerald-600 dark:text-emerald-300">传给 UE</div>
          <div className="mt-1 truncate font-medium text-emerald-800 dark:text-emerald-200">
            {ueAction}
          </div>
        </div>
      </div>
    </div>
  );
}
