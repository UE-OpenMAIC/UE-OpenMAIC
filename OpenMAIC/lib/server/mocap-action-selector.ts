import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { nanoid } from 'nanoid';

import { getMocapMapping } from '@/lib/mocap/action-map';
import type { Action, MocapAction, SpeechAction } from '@/lib/types/action';
import { createLogger } from '@/lib/logger';

const log = createLogger('MocapActionSelector');

const SELECTOR_DIR = process.env.MOCAP_SELECTOR_DIR || 'D:\\code\\teacherT2S\\Time2State\\llm';
const SELECTOR_SCRIPT = path.join(SELECTOR_DIR, 'rag_mocap_action_selector.py');
const RAG_DIR = process.env.MOCAP_RAG_DIR || path.join(SELECTOR_DIR, 'mocap_action_rag');
const PYTHON_BIN = process.env.MOCAP_SELECTOR_PYTHON || 'python';
const FORCE_DEEPSEEK_SELECTOR =
  (process.env.MOCAP_SELECTOR_CHAT_MODEL || '').toLowerCase().includes('deepseek') ||
  (process.env.MOCAP_SELECTOR_OPENAI_BASE_URL || '').toLowerCase().includes('deepseek');
const SELECTOR_OPENAI_API_KEY =
  process.env.MOCAP_SELECTOR_OPENAI_API_KEY ||
  process.env.DEEPSEEK_API_KEY ||
  (!FORCE_DEEPSEEK_SELECTOR ? process.env.OPENAI_API_KEY : undefined);
const SELECTOR_OPENAI_BASE_URL =
  process.env.MOCAP_SELECTOR_OPENAI_BASE_URL ||
  (FORCE_DEEPSEEK_SELECTOR || process.env.DEEPSEEK_API_KEY
    ? process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com'
    : process.env.OPENAI_BASE_URL || process.env.OPENAI_API_BASE_URL);
const USE_CHAT =
  process.env.MOCAP_SELECTOR_USE_CHAT === 'true' ||
  (process.env.MOCAP_SELECTOR_USE_CHAT !== 'false' && !!SELECTOR_OPENAI_API_KEY);
const CHAT_MODEL =
  process.env.MOCAP_SELECTOR_CHAT_MODEL ||
  process.env.CHAT_MODEL ||
  (process.env.DEEPSEEK_API_KEY ? 'deepseek-chat' : undefined);

type SelectorPlanItem = {
  sentence_id?: number;
  sentence?: string;
  recommended_mocap_id?: string;
  recommended_mocap_action?: string;
  recommended_mocap_coarse?: string;
  confidence_score?: number;
};

function selectMocapForTexts(texts: string[]): SelectorPlanItem[] {
  if (texts.length === 0) return [];

  const tempDir = mkdtempSync(path.join(tmpdir(), 'openmaic-mocap-'));
  const textFile = path.join(tempDir, 'speech.txt');
  const outDir = path.join(tempDir, 'out');

  try {
    writeFileSync(textFile, texts.join('\n'), 'utf8');

    const runSelector = (useChat: boolean) => {
      const args = [
        '-X',
        'utf8',
        SELECTOR_SCRIPT,
        'plan',
        '--text-file',
        textFile,
        '--sentence-only',
        '--top-k',
        '20',
        '--rag-dir',
        RAG_DIR,
        '--out-dir',
        outDir,
      ];
      if (useChat) {
        args.push('--use-chat');
        if (CHAT_MODEL) {
          args.push('--chat-model', CHAT_MODEL);
        }
      }

      log.info(
        `Running mocap selector: mode=${useChat ? 'rag+llm' : 'rag'}, texts=${texts.length}, model=${CHAT_MODEL || 'default'}`,
      );

      const result = spawnSync(PYTHON_BIN, args, {
        cwd: SELECTOR_DIR,
        encoding: 'utf8',
        env: {
          ...process.env,
          OPENAI_API_KEY: SELECTOR_OPENAI_API_KEY,
          OPENAI_BASE_URL: SELECTOR_OPENAI_BASE_URL,
          PYTHONUTF8: '1',
          PYTHONIOENCODING: 'utf-8',
        },
        timeout: 30_000,
      });

      if (result.error) {
        throw result.error;
      }
      if (result.status !== 0) {
        throw new Error(result.stderr || result.stdout || `selector exited ${result.status}`);
      }

      const chatOutputPath = path.join(outDir, 'last_mocap_action_plan_chat.json');
      const deterministicOutputPath = path.join(
        outDir,
        'last_mocap_action_plan_deterministic.json',
      );
      const outputPath =
        useChat && exists(chatOutputPath) ? chatOutputPath : deterministicOutputPath;
      const plan = parseSelectorJson(readFileSync(outputPath, 'utf8'));
      log.info(
        `Mocap selector result: mode=${useChat ? 'rag+llm' : 'rag'}, ids=${plan.map((item) => item.recommended_mocap_id || 'none').join(', ')}`,
      );
      return plan;
    };

    if (!USE_CHAT) {
      return runSelector(false);
    }

    try {
      return runSelector(true);
    } catch (error) {
      log.warn('Mocap selector rag+llm failed; retrying local rag selector:', error);
      return runSelector(false);
    }
  } finally {
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch {
      // temp cleanup best-effort
    }
  }
}

function exists(filePath: string): boolean {
  try {
    readFileSync(filePath);
    return true;
  } catch {
    return false;
  }
}

function parseSelectorJson(raw: string): SelectorPlanItem[] {
  const trimmed = raw.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim();
  const candidate = fenced || trimmed;
  const parsed = JSON.parse(candidate) as SelectorPlanItem[];
  return Array.isArray(parsed) ? parsed : [];
}

function toMocapAction(planItem: SelectorPlanItem | undefined, sourceText: string): MocapAction {
  const mocapId = planItem?.recommended_mocap_id || 'MC01';
  const mapping = getMocapMapping(mocapId);
  if (!planItem?.recommended_mocap_id) {
    log.warn(`Mocap plan missing for speech, falling back to MC01: ${sourceText}`);
  }

  return {
    id: `mocap_${nanoid(8)}`,
    type: 'mocap',
    mocapId,
    mocapAction: planItem?.recommended_mocap_action || mapping?.mocapAction || '',
    mocapCoarse: planItem?.recommended_mocap_coarse || mapping?.mocapCoarse || '',
    confidenceScore: planItem?.confidence_score,
    sourceText,
    teacherStatus: mapping?.teacherStatus || 'Talking',
    ueAction: mapping?.ueAction || 'Talking',
  };
}

function forceBoardWritingMocap(sourceText: string): MocapAction {
  const mapping = getMocapMapping('MC12');
  return {
    id: `mocap_${nanoid(8)}`,
    type: 'mocap',
    mocapId: 'MC12',
    mocapAction: mapping?.mocapAction || '常规板书书写',
    mocapCoarse: mapping?.mocapCoarse || '板书书写',
    sourceText,
    teacherStatus: mapping?.teacherStatus || 'MC12',
    ueAction: mapping?.ueAction || 'MC12',
  };
}

function isBoardWritingMocap(mocapId: string): boolean {
  return mocapId === 'MC12' || mocapId === 'MC13' || mocapId === 'MC14' || mocapId === 'MC15';
}

function stripWriteFields(action: SpeechAction): SpeechAction {
  const {
    writeElementId: _writeElementId,
    writeContent: _writeContent,
    writeFontSize: _writeFontSize,
    writeColor: _writeColor,
    ...rest
  } = action;
  return rest;
}

export function enrichActionsWithMocap(actions: Action[]): Action[] {
  const speechActions = actions.filter(
    (action): action is SpeechAction => action.type === 'speech',
  );
  if (speechActions.length === 0) return actions;

  let plan: SelectorPlanItem[] = [];
  try {
    plan = selectMocapForTexts(
      speechActions.map((action) => action.mocapSelectorText || action.text),
    );
  } catch (error) {
    log.warn('Mocap selector failed; falling back to MC01:', error);
  }

  let speechIndex = 0;
  const enriched: Action[] = [];
  for (const action of actions) {
    if (action.type === 'speech') {
      const planItem = plan[speechIndex];
      const selectedMocapAction = toMocapAction(planItem, action.text);
      const mocapAction =
        action.writeElementId && !isBoardWritingMocap(selectedMocapAction.mocapId)
          ? forceBoardWritingMocap(action.text)
          : selectedMocapAction;
      enriched.push(mocapAction);
      speechIndex += 1;

      if (action.writeElementId && !isBoardWritingMocap(mocapAction.mocapId)) {
        enriched.push(stripWriteFields(action));
        continue;
      }
    }
    enriched.push(action);
  }

  return enriched;
}
