export type BaseTeacherStatus = 'Idle' | 'Talking' | 'Pointing' | 'Thinking';
export type UEMocapStatus =
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
export type TeacherStatus = BaseTeacherStatus | UEMocapStatus;

export type MocapId =
  | 'MC01'
  | 'MC02'
  | 'MC03'
  | 'MC04'
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

export type MocapMapping = {
  mocapId: MocapId;
  mocapAction: string;
  mocapCoarse: string;
  teacherStatus: TeacherStatus;
  ueAction: string;
};

export const MOCAP_ACTION_MAPPINGS: Record<MocapId, MocapMapping> = {
  MC01: {
    mocapId: 'MC01',
    mocapAction: '正向静立/双手腹前讲解',
    mocapCoarse: '正向讲授',
    teacherStatus: 'Talking',
    ueAction: 'Talking',
  },
  MC02: {
    mocapId: 'MC02',
    mocapAction: '正向单手指向屏幕',
    mocapCoarse: '正向讲授',
    teacherStatus: 'Pointing',
    ueAction: 'Pointing',
  },
  MC03: {
    mocapId: 'MC03',
    mocapAction: '正向前伸/推掌/摊开讲解',
    mocapCoarse: '正向讲授',
    teacherStatus: 'Talking',
    ueAction: 'Talking',
  },
  MC04: {
    mocapId: 'MC04',
    mocapAction: '侧身/斜侧静立讲解',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'Pointing',
    ueAction: 'Pointing',
  },
  MC05: {
    mocapId: 'MC05',
    mocapAction: '侧身双手腹前讲解',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'MC05',
    ueAction: 'MC05',
  },
  MC06: {
    mocapId: 'MC06',
    mocapAction: '侧身抱臂/夹臂/自然下垂',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'MC06',
    ueAction: 'MC06',
  },
  MC07: {
    mocapId: 'MC07',
    mocapAction: '侧身/斜侧指向屏幕',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'MC07',
    ueAction: 'MC07',
  },
  MC08: {
    mocapId: 'MC08',
    mocapAction: '侧身高位/上举指向',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'MC08',
    ueAction: 'MC08',
  },
  MC09: {
    mocapId: 'MC09',
    mocapAction: '侧身向前伸手/挥掌',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'MC09',
    ueAction: 'MC09',
  },
  MC10: {
    mocapId: 'MC10',
    mocapAction: '屏幕/PPT操作',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'MC10',
    ueAction: 'MC10',
  },
  MC11: {
    mocapId: 'MC11',
    mocapAction: '正侧身转向过渡',
    mocapCoarse: '侧向讲授',
    teacherStatus: 'MC11',
    ueAction: 'MC11',
  },
  MC12: {
    mocapId: 'MC12',
    mocapAction: '常规板书书写',
    mocapCoarse: '板书书写',
    teacherStatus: 'MC12',
    ueAction: 'MC12',
  },
  MC13: {
    mocapId: 'MC13',
    mocapAction: '板书上下方向书写',
    mocapCoarse: '板书书写',
    teacherStatus: 'MC13',
    ueAction: 'MC13',
  },
  MC14: {
    mocapId: 'MC14',
    mocapAction: '板书后转身/边写边讲',
    mocapCoarse: '板书书写',
    teacherStatus: 'MC14',
    ueAction: 'MC14',
  },
  MC15: {
    mocapId: 'MC15',
    mocapAction: '斜侧/背身/遮挡板书',
    mocapCoarse: '板书书写',
    teacherStatus: 'MC15',
    ueAction: 'MC15',
  },
};

export function getMocapMapping(mocapId?: string): MocapMapping | null {
  if (!mocapId || !(mocapId in MOCAP_ACTION_MAPPINGS)) return null;
  return MOCAP_ACTION_MAPPINGS[mocapId as MocapId];
}
