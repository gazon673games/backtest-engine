import fs from "node:fs";
import readline from "node:readline";
import vm from "node:vm";

const strategyPath = process.argv[2];
if (!strategyPath) {
  throw new Error("strategy path is required");
}

const plots = [];
const logs = [];
let syncedHistory = { columns: {}, length: 0 };
let syncedSession = { columns: {}, length: 0 };
let strategyState = {};
const sandbox = {
  Array,
  Boolean,
  Date,
  JSON,
  Math,
  Number,
  Object,
  String,
  console: {
    error: (...args) => pushLog("error", args),
    log: (...args) => pushLog("log", args),
  },
  math: {
    avg,
    ema,
    max: (values) => Math.max(...clean(values)),
    min: (values) => Math.min(...clean(values)),
    sma,
    sum,
  },
  plot: {
    line: (name, value) => plots.push({ type: "line", name, value }),
    marker: (name, value) => plots.push({ type: "marker", name, value }),
    panel: (name, value) => plots.push({ type: "panel", name, value }),
  },
};

const context = vm.createContext(sandbox, {
  codeGeneration: { strings: false, wasm: false },
});
const source = fs.readFileSync(strategyPath, "utf8");
new vm.Script(source, { filename: strategyPath }).runInContext(context, { timeout: 1000 });

sandbox.__invoke = () => {
  plots.length = 0;
  logs.length = 0;
  const handler = sandbox.onDecision || sandbox.strategy?.onDecision;
  if (typeof handler !== "function") {
    throw new Error("JS strategy must define onDecision(ctx, params)");
  }
  const result = handler(buildCtx(sandbox.__ctx), sandbox.__params || {});
  if (result && typeof result === "object" && !Array.isArray(result) && plots.length > 0) {
    return withDiagnostics({ ...result, plots: result.plots || plots.slice() });
  }
  return withDiagnostics(result);
};

write({ ready: true });
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) {
    continue;
  }
  const request = JSON.parse(line);
  try {
    const result = dispatch(request, context);
    write({ id: request.id, ok: true, result });
  } catch (error) {
    write({
      id: request.id,
      ok: false,
      error: String(error && error.stack ? error.stack : error),
    });
  }
}

function dispatch(request, context) {
  if (request.type === "metadata") return metadata();
  if (request.type === "start") return startLifecycle(request);
  if (request.type === "fill") return lifecycleHook("onFill", request.event || {});
  if (request.type === "end") return lifecycleHook("onEnd", request.summary || {});
  return executeDecision(request, context);
}

function startLifecycle(request) {
  const handler = sandbox.initialize || sandbox.onStart || sandbox.strategy?.initialize;
  strategyState = {};
  if (typeof handler === "function") {
    const initialized = handler(request.params || {});
    if (initialized !== undefined && initialized !== null) {
      if (typeof initialized !== "object" || Array.isArray(initialized)) {
        throw new Error("JS initialize must return an object or null");
      }
      strategyState = initialized;
    }
  }
  return null;
}

function lifecycleHook(name, payload) {
  const handler = sandbox[name] || sandbox.strategy?.[name];
  if (typeof handler === "function") handler(strategyState, payload);
  return null;
}

function executeDecision(request, context) {
  sandbox.__ctx = { ...applyContextSync(request.ctx || {}), state: strategyState };
  sandbox.__params = request.params || {};
  return vm.runInContext("__invoke()", context, {
    timeout: Number(request.timeoutMs || 1000),
  });
}

function applyContextSync(raw) {
  if (!raw.historySync && !raw.sessionSync) {
    return raw;
  }
  syncedHistory = mergeSource(syncedHistory, raw.historySync, raw.historySync?.limit);
  syncedSession = mergeSource(syncedSession, raw.sessionSync);
  const { historySync, sessionSync, ...scalars } = raw;
  return { ...scalars, history: syncedHistory, session: syncedSession };
}

function mergeSource(current, sync, limit = null) {
  if (!sync) {
    return current;
  }
  const data = sync.data || { columns: {}, length: 0 };
  const previousLength = sync.reset ? 0 : Number(current.length || 0);
  const columns = {};
  for (const [name, values] of Object.entries(data.columns || {})) {
    const previous = sync.reset ? [] : current.columns?.[name] || [];
    columns[name] = previous.concat(values);
    if (limit && columns[name].length > limit) {
      columns[name] = columns[name].slice(-limit);
    }
  }
  const mergedLength = previousLength + Number(data.length || 0);
  return { columns, length: limit ? Math.min(limit, mergedLength) : mergedLength };
}

function metadata() {
  return sandbox.STRATEGY_METADATA || sandbox.strategy?.metadata || null;
}

function write(payload) {
  process.stdout.write(JSON.stringify(payload) + "\n");
}

function pushLog(level, args) {
  if (logs.length >= 100) {
    return;
  }
  logs.push({ level, message: args.map(String).join(" ") });
}

function withDiagnostics(result) {
  if (!result || typeof result !== "object" || Array.isArray(result) || logs.length === 0) {
    return result;
  }
  return { ...result, logs: result.logs || logs.slice() };
}

function buildCtx(raw) {
  return {
    ...raw,
    indicators: {
      ema: (column, window, source = "history") => ema(valuesFromSource(raw, source, column), window),
      sma: (column, window, source = "history") => sma(valuesFromSource(raw, source, column), window),
    },
    history: sourceApi(raw.history),
    session: sourceApi(raw.session),
  };
}

function sourceApi(source) {
  const safeSource = source || { columns: {}, length: 0 };
  return {
    columns: safeSource.columns || {},
    length: safeSource.length || 0,
    rows: () => rowsFromColumns(safeSource.columns || {}, safeSource.length || 0),
    value: (column, offset = -1) => valueAt(columnValues(safeSource, column), offset),
    values: (column, options = {}) => selectValues(safeSource, column, options),
  };
}

function valuesFromSource(raw, source, column) {
  const selected = source === "session" ? raw.session : raw.history;
  return selectValues(selected || {}, column, {});
}

function rowsFromColumns(columns, length) {
  const rows = [];
  for (let index = 0; index < length; index += 1) {
    const row = {};
    for (const [name, values] of Object.entries(columns)) {
      row[name] = values[index];
    }
    rows.push(row);
  }
  return rows;
}

function valueAt(values, offset) {
  const index = offset < 0 ? values.length + offset : offset;
  return index < 0 || index >= values.length ? null : values[index];
}

function selectValues(source, column, options) {
  const values = columnValues(source, column);
  const times = options.from || options.to ? columnValues(source, "ts") : null;
  const from = options.from ? String(options.from) : null;
  const to = options.to ? String(options.to) : null;
  const selected = values.filter((_, index) => {
    const time = times ? timePart(times[index]) : "";
    if (from && time < from) return false;
    return !to || time <= to;
  });
  if (options.bars || options.fromBarsAgo) {
    return selected.slice(-Number(options.bars || options.fromBarsAgo));
  }
  return selected;
}

function columnValues(source, column) {
  const columns = source.columns || {};
  if (!Object.prototype.hasOwnProperty.call(columns, column)) {
    throw new Error(`Unknown strategy data column: ${column}`);
  }
  return columns[column];
}

function timePart(value) {
  if (typeof value !== "string") {
    return "";
  }
  const match = value.match(/T?(\d\d:\d\d)/);
  return match ? match[1] : value.slice(0, 5);
}

function clean(values) {
  return (values || []).filter((value) => value !== null && Number.isFinite(Number(value)));
}

function sum(values) {
  return clean(values).reduce((total, value) => total + Number(value), 0);
}

function avg(values) {
  const valid = clean(values);
  return valid.length === 0 ? null : sum(valid) / valid.length;
}

function sma(values, window) {
  const valid = clean(values).slice(-Number(window));
  return valid.length < Number(window) ? null : avg(valid);
}

function ema(values, window) {
  const valid = clean(values);
  const period = Number(window);
  if (valid.length < period) {
    return null;
  }
  const alpha = 2 / (period + 1);
  let value = avg(valid.slice(0, period));
  for (const next of valid.slice(period)) {
    value = Number(next) * alpha + value * (1 - alpha);
  }
  return value;
}
