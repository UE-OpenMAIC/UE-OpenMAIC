'use client';

export type TeacherStatus =
  | 'Idle'
  | 'Talking'
  | 'Pointing'
  | 'Thinking'
  | 'MC05'
  | 'MC06'
  | 'MC07'
  | 'MC08'
  | 'MC09'
  | 'MC10'
  | 'MC11'
  | 'MC12'
  | 'MC13'
  | 'MC14'
  | 'MC15';

type TeacherPanelProps = {
  teacherName?: string;
  teacherRole?: string;
  teacherAvatar?: string;
  status?: TeacherStatus;
};

export function TeacherPanel({
  teacherName = '王老师',
  teacherRole = '主讲教师',
  teacherAvatar = '/avatars/teacher.png',
  status = 'Idle',
}: TeacherPanelProps) {
  const statusMap: Record<TeacherStatus, string> = {
    Idle: '待命中',
    Talking: '讲解中',
    Pointing: '指向重点',
    Thinking: '思考中',
    MC05: 'MC05',
    MC06: 'MC06',
    MC07: 'MC07',
    MC08: 'MC08',
    MC09: 'MC09',
    MC10: 'MC10',
    MC11: 'MC11',
    MC12: 'MC12',
    MC13: 'MC13',
    MC14: 'MC14',
    MC15: 'MC15',
  };

  const panelStyleMap: Record<TeacherStatus, string> = {
    Idle: 'border-slate-200/70 dark:border-slate-700/70',
    Talking: 'border-blue-300/70 dark:border-blue-700/70',
    Pointing: 'border-amber-300/70 dark:border-amber-700/70',
    Thinking: 'border-purple-300/70 dark:border-purple-700/70',
    MC05: 'border-blue-300/70 dark:border-blue-700/70',
    MC06: 'border-purple-300/70 dark:border-purple-700/70',
    MC07: 'border-amber-300/70 dark:border-amber-700/70',
    MC08: 'border-amber-300/70 dark:border-amber-700/70',
    MC09: 'border-amber-300/70 dark:border-amber-700/70',
    MC10: 'border-amber-300/70 dark:border-amber-700/70',
    MC11: 'border-amber-300/70 dark:border-amber-700/70',
    MC12: 'border-emerald-300/70 dark:border-emerald-700/70',
    MC13: 'border-emerald-300/70 dark:border-emerald-700/70',
    MC14: 'border-emerald-300/70 dark:border-emerald-700/70',
    MC15: 'border-emerald-300/70 dark:border-emerald-700/70',
  };

  const statusBoxStyleMap: Record<TeacherStatus, string> = {
    Idle: 'bg-slate-100 dark:bg-slate-800 text-muted-foreground',
    Talking: 'bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300',
    Pointing: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300',
    Thinking: 'bg-purple-50 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300',
    MC05: 'bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300',
    MC06: 'bg-purple-50 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300',
    MC07: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300',
    MC08: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300',
    MC09: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300',
    MC10: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300',
    MC11: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300',
    MC12: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300',
    MC13: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300',
    MC14: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300',
    MC15: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300',
  };

  return (
    <div
      className={`w-full rounded-2xl border bg-white/80 dark:bg-slate-900/70 backdrop-blur p-4 shadow-sm ${panelStyleMap[status]}`}
    >
      {/* 面板标题 */}
      <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">教师形象面板</div>

      {/* 老师头像 */}
      <div className="mt-4 flex justify-center">
        <img
          src={teacherAvatar}
          alt={teacherName}
          className="h-48 w-auto object-contain rounded-xl"
        />
      </div>

      {/* 老师信息 */}
      <div className="mt-4 text-center">
        <div className="text-base font-medium text-slate-900 dark:text-slate-100">
          {teacherName}
        </div>
        <div className="mt-1 text-sm text-muted-foreground">{teacherRole}</div>
      </div>

      {/* 动态状态显示 */}
      <div
        className={`mt-4 rounded-lg px-3 py-2 text-xs text-center font-medium ${statusBoxStyleMap[status]}`}
      >
        当前状态：{statusMap[status]}
      </div>
    </div>
  );
}
