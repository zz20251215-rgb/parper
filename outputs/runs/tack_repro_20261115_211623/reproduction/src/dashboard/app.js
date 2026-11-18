'use strict';

const $ = (id) => document.getElementById(id);
const COLORS = {blue:'#0072B2', orange:'#D55E00', teal:'#009E73', purple:'#CC79A7', amber:'#E69F00', slate:'#475569'};
const STATUS = {paused:'已暂停', running:'训练中', completed:'已完成', failed:'运行失败', stopped:'已停止', unknown:'状态未知'};
let current = null;
let refreshing = false;

function number(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
function fixed(value, digits = 3) {
  const n = number(value);
  return n === null ? '—' : n.toFixed(digits);
}
function integer(value) {
  const n = number(value);
  return n === null ? '—' : n.toLocaleString('zh-CN', {maximumFractionDigits:0});
}
function percent(value, digits = 1) {
  const n = number(value);
  return n === null ? '—' : `${(n * 100).toFixed(digits)}%`;
}
function temperature(value) {
  const n = number(value);
  return n === null ? '—' : `${n.toFixed(1)} °C`;
}
function gib(value) {
  const n = number(value);
  return n === null ? '—' : `${(n / 1024).toFixed(2)} GiB`;
}
function time(value, full = false) {
  const date = value ? new Date(value) : null;
  if (!date || !Number.isFinite(date.getTime())) return '—';
  return date.toLocaleTimeString('zh-CN', {hour12:false, hour:'2-digit', minute:'2-digit', ...(full ? {second:'2-digit'} : {})});
}
function text(id, value) { $(id).textContent = value; }
function bar(id, value) {
  $(id).style.width = `${Math.max(0, Math.min(100, number(value) ?? 0))}%`;
}
function svgElement(name, attrs = {}, content) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  if (content !== undefined) el.textContent = content;
  return el;
}
function marker(x, y, item, size = 3.2) {
  const attrs = {fill:'white', stroke:item.color, 'stroke-width':1.5, 'vector-effect':'non-scaling-stroke'};
  if (item.marker === 'square') return svgElement('rect', {...attrs,x:x-size,y:y-size,width:size*2,height:size*2});
  if (item.marker === 'triangle') return svgElement('polygon', {...attrs,points:`${x},${y-size-1} ${x+size+1},${y+size} ${x-size-1},${y+size}`});
  if (item.marker === 'diamond') return svgElement('polygon', {...attrs,points:`${x},${y-size-1} ${x+size+1},${y} ${x},${y+size+1} ${x-size-1},${y}`});
  return svgElement('circle', {...attrs,cx:x,cy:y,r:size});
}
function legend(id, series) {
  const parent = $(id);
  parent.replaceChildren();
  for (const item of series) {
    const span = document.createElement('span');
    const sample = svgElement('svg', {viewBox:'0 0 40 14',width:40,height:14,'aria-hidden':'true'});
    sample.append(svgElement('line', {x1:1,x2:39,y1:7,y2:7,stroke:item.color,'stroke-width':2,'stroke-dasharray':item.dash || 'none'}));
    sample.append(marker(20,7,item,2.5));
    span.append(sample, document.createTextNode(item.name));
    parent.append(span);
  }
}
function drawChart(id, rows, series, options = {}) {
  const parent = $(id);
  parent.replaceChildren();
  const read = (item, row) => {
    const value = number(item.get(row));
    return options.log && value !== null && value <= 0 ? null : value;
  };
  const values = rows.flatMap((row) => series.map((item) => read(item,row))).filter((v) => v !== null);
  if (!rows.length || !values.length) {
    parent.classList.add('empty');
    parent.textContent = options.empty || '暂无已记录的数据';
    return;
  }
  parent.classList.remove('empty');
  const width = Math.max(parent.clientWidth || 640, 300), height = parent.clientHeight || 240;
  const pad = {left:67,right:19,top:28,bottom:45};
  let min = options.min ?? (options.log ? Math.min(...values)/2 : Math.min(0,...values));
  let max = options.max ?? Math.max(...values)*1.12;
  if (max <= min) max = min + (options.log ? min : 1);
  const transform = options.log ? Math.log10 : (value) => value;
  const low = transform(min), high = transform(max);
  const chartW = width-pad.left-pad.right, chartH = height-pad.top-pad.bottom;
  const xValues = rows.map((row,i) => number(options.xValue ? options.xValue(row) : row.epoch) ?? i);
  const firstX = Math.min(...xValues), lastX = Math.max(...xValues);
  const x = (i) => pad.left+(lastX===firstX ? chartW/2 : (xValues[i]-firstX)/(lastX-firstX)*chartW);
  const y = (value) => pad.top+(high-transform(value))/(high-low)*chartH;
  const svg = svgElement('svg', {viewBox:`0 0 ${width} ${height}`,preserveAspectRatio:'xMidYMid meet','aria-hidden':'true'});
  svg.append(svgElement('text', {x:pad.left,y:13,fill:'#475569','font-size':11},options.yLabel || 'Value'));
  const ticks = options.log ? Array.from({length:Math.floor(high)-Math.ceil(low)+1},(_,i)=>10**(Math.ceil(low)+i))
    : Array.from({length:5},(_,i)=>min+(max-min)*i/4);
  const format = options.tickFormat || ((value)=>value.toFixed(options.digits ?? 2));
  for (const value of ticks) {
    const yy = y(value);
    svg.append(svgElement('line', {x1:pad.left,x2:width-pad.right,y1:yy,y2:yy,stroke:'#E2E8F0','stroke-width':1,'stroke-dasharray':'2 4'}));
    svg.append(svgElement('text', {x:pad.left-10,y:yy+4,fill:'#64748B','font-size':11,'text-anchor':'end'},format(value)));
  }
  svg.append(svgElement('path', {d:`M${pad.left},${pad.top}V${height-pad.bottom}H${width-pad.right}`,fill:'none',stroke:'#94A3B8','stroke-width':1}));
  for (const i of new Set([0,Math.floor((rows.length-1)/2),rows.length-1])) {
    svg.append(svgElement('text', {x:x(i),y:height-pad.bottom+19,fill:'#64748B','font-size':11,'text-anchor':'middle'},options.label ? options.label(rows[i],i) : String(rows[i].epoch)));
  }
  svg.append(svgElement('text', {x:pad.left+chartW/2,y:height-3,fill:'#475569','font-size':11,'text-anchor':'middle'},options.xLabel || 'Epoch'));
  for (const item of series) {
    let d = '', contiguous = false;
    rows.forEach((row,i) => {
      const value = read(item,row);
      if (value === null) {contiguous=false;return;}
      d += contiguous ? (item.step ? `H${x(i).toFixed(2)}V${y(value).toFixed(2)} ` : `L${x(i).toFixed(2)},${y(value).toFixed(2)} `) : `M${x(i).toFixed(2)},${y(value).toFixed(2)} `;
      contiguous = true;
    });
    svg.append(svgElement('path', {d,fill:'none',stroke:item.color,'stroke-width':2,'stroke-dasharray':item.dash || 'none','stroke-linecap':'round','stroke-linejoin':'round','vector-effect':'non-scaling-stroke'}));
    rows.forEach((row,i) => {
      const value = read(item,row);
      if (value === null || (rows.length>30 && i%Math.ceil(rows.length/20)!==0 && i!==rows.length-1)) return;
      const point = marker(x(i),y(value),item);
      const detail = options.valueFormat ? options.valueFormat(value) : value.toFixed(5);
      point.append(svgElement('title',{},`${options.label ? options.label(row,i) : `Epoch ${row.epoch}`} · ${item.name}: ${detail}`));
      svg.append(point);
    });
  }
  parent.append(svg);
}
function trainingCharts(history, cfg) {
  const charts = [
    ['loss', [{name:'训练 Loss',color:COLORS.blue,marker:'circle',get:r=>r.train_loss},{name:'验证 Loss',color:COLORS.orange,dash:'7 4',marker:'square',get:r=>r.val?.loss}], {min:0,yLabel:'Loss'}],
    ['metric', [{name:'Dice / F1',color:COLORS.blue,marker:'circle',get:r=>r.val?.dice},{name:'IoU',color:COLORS.orange,dash:'7 4',marker:'square',get:r=>r.val?.iou}], {min:0,max:1,yLabel:'Score'}],
    ['pr', [{name:'Precision',color:COLORS.teal,marker:'triangle',get:r=>r.val?.precision},{name:'Recall',color:COLORS.purple,dash:'8 3 2 3',marker:'diamond',get:r=>r.val?.recall}], {min:0,max:1,yLabel:'Score'}],
    ['lr', [{name:'本轮学习率',color:COLORS.slate,marker:'square',step:true,get:r=>r.learning_rate}], {log:true,min:number(cfg.minimum_lr) || 1e-5,max:number(cfg.learning_rate) || 1e-3,yLabel:'Learning rate',tickFormat:v=>v.toExponential(0),valueFormat:v=>v.toExponential(3)}],
    ['negative-image', [{name:'图片误报率',color:COLORS.orange,dash:'7 4',marker:'square',get:r=>r.val?.negative_image_fp_rate}], {min:0,max:1,yLabel:'False positive (%)',tickFormat:v=>`${(v*100).toFixed(0)}%`,valueFormat:v=>`${(v*100).toFixed(3)}%`}],
    ['negative-pixel', [{name:'像素误报率',color:COLORS.purple,dash:'8 3 2 3',marker:'diamond',get:r=>r.val?.negative_pixel_fp_rate}], {min:0,yLabel:'False positive (%)',tickFormat:v=>`${(v*100).toFixed(3)}%`,valueFormat:v=>`${(v*100).toFixed(5)}%`}]
  ];
  for (const [id,series,options] of charts) {
    legend(`${id}-legend`,series);
    drawChart(`${id}-chart`,history,series,options);
  }
}
function renderResources(state) {
  const resource = state.resources || {};
  const gpu = resource.gpu || {}, ram = resource.ram || {}, cpu = resource.cpu || {};
  text('gpu-name', gpu.name || 'GPU 信息暂不可用');
  text('sensor-updated', `采样 ${time(resource.timestamp, true)}`);
  text('gpu-memory', gib(gpu.used_mb));
  text('gpu-memory-total', `总量 ${gib(gpu.total_mb)}`);
  const gpuPct = number(gpu.used_mb) !== null && number(gpu.total_mb) > 0 ? gpu.used_mb / gpu.total_mb * 100 : null;
  bar('gpu-memory-bar', gpuPct);
  text('gpu-util', `利用率 ${fixed(gpu.utilization_pct, 0)}${number(gpu.utilization_pct) === null ? '' : '%'}`);
  text('ram-memory', gib(ram.used_mb));
  text('ram-total', `总量 ${gib(ram.total_mb)}`);
  const ramPct = number(ram.percent) ?? (number(ram.total_mb) > 0 ? ram.used_mb / ram.total_mb * 100 : null);
  bar('ram-memory-bar', ramPct);
  text('process-memory', `训练进程 ${gib(resource.process?.rss_mb)}`);
  text('gpu-temperature', temperature(gpu.temperature_c));
  text('cpu-temperature', temperature(cpu.temperature_c));
  text('gpu-power', `功耗 ${fixed(gpu.power_w, 1)}${number(gpu.power_w) === null ? '' : ' W'}`);
  text('cpu-util', `CPU 利用率 ${fixed(cpu.utilization_pct, 0)}${number(cpu.utilization_pct) === null ? '' : '%'}`);
  const errors = (resource.errors || []).map(String);
  $('sensor-errors').hidden = errors.length === 0;
  text('sensor-errors', errors.length ? `部分传感器暂不可用：${errors.join('；')}` : '');
  renderResourceChart(state);
}
function renderResourceChart(state) {
  const rows = Array.isArray(state.resource_history) ? state.resource_history : [];
  let series, max, digits;
  const mode = $('resource-mode').value;
  if (mode === 'temperature') {
    series = [{name:'GPU 温度 / °C', color:COLORS.teal, get:r=>r.gpu?.temperature_c}, {name:'CPU 温度 / °C', color:COLORS.orange, dash:'7 4', marker:'square', get:r=>r.cpu?.temperature_c}];
    max = Math.max(60, ...rows.flatMap(r=>[number(r.gpu?.temperature_c),number(r.cpu?.temperature_c)]).filter(v=>v !== null)) * 1.1;
    digits = 0;
  } else if (mode === 'utilization') {
    series = [{name:'GPU 利用率 / %', color:COLORS.teal, get:r=>r.gpu?.utilization_pct}, {name:'CPU 利用率 / %', color:COLORS.orange, dash:'7 4', marker:'square', get:r=>r.cpu?.utilization_pct}];
    max = 100; digits = 0;
  } else {
    series = [{name:'GPU 显存占用 / %', color:COLORS.teal, get:r=>number(r.gpu?.total_mb)>0 ? r.gpu.used_mb/r.gpu.total_mb*100 : null}, {name:'系统内存占用 / %', color:COLORS.orange, dash:'7 4', marker:'square', get:r=>r.ram?.percent}];
    max = 100; digits = 0;
  }
  legend('resource-legend', series);
  drawChart('resource-chart', rows, series, {min:0, max, digits, xValue:r=>new Date(r.timestamp).getTime(), xLabel:'Time', yLabel:mode==='temperature' ? 'Temperature (°C)' : 'Usage (%)', label:r=>time(r.timestamp), empty:'尚无监测记录，开始采样后将在此显示真实趋势'});
}
function renderConfig(state, latest) {
  const cfg = state.config || {}, early = cfg.early_stopping || {};
  const list = $('config-list');
  list.replaceChildren();
  const lr = number(latest?.next_learning_rate ?? latest?.learning_rate ?? cfg.learning_rate);
  const values = [
    ['模型', cfg.model || '—'], ['初始化', cfg.pretrained === false ? '从随机权重训练' : cfg.pretrained === true ? '使用预训练权重' : '—'],
    ['当前学习率', lr === null ? '—' : lr.toExponential(2)], ['优化器 / 调度', [cfg.optimizer,cfg.scheduler].filter(Boolean).join(' / ') || '—'],
    ['Batch size', integer(cfg.batch_size)], ['随机种子', integer(cfg.seed)],
    ['早停耐心', number(early.patience) === null ? '—' : `${integer(latest?.epochs_without_improvement ?? 0)} / ${integer(early.patience)} 轮`],
    ['最少训练轮数', integer(early.min_epochs)]
  ];
  for (const [label,value] of values) {
    const div = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = label; dd.textContent = value; div.append(dt,dd); list.append(div);
  }
  const repro = state.reproducibility;
  const complete = repro && (repro.status === 'verified' || repro.deterministic === true || repro.deterministic_algorithms === true);
  $('repro-note').className = `notice ${complete ? 'success' : 'warning'}`;
  let note = repro?.message || repro?.note;
  if (!note && Array.isArray(repro?.limitations)) note = repro.limitations.join('；');
  if (!note && typeof repro?.limitations === 'string') note = repro.limitations;
  text('repro-note', note || (complete ? '已记录确定性设置与复现信息。相同软件、硬件和数据环境下复现。' : '此运行启动时未完整记录确定性设置及随机、优化器状态，无法事后保证精确复现。'));
}
function renderData(state, latest) {
  const summary = state.config?.split_summary || {}, body = $('split-table');
  body.replaceChildren();
  for (const [key,name] of [['train','训练'],['val','验证'],['test','测试']]) {
    const item = summary[key] || {}, row = document.createElement('tr');
    for (const value of [name,integer(item.images),integer(item.positive),integer(item.negative)]) {
      const td = document.createElement('td'); td.textContent = value; row.append(td);
    }
    body.append(row);
  }
  text('negative-image-fp', percent(latest?.val?.negative_image_fp_rate));
  text('negative-pixel-fp', percent(latest?.val?.negative_pixel_fp_rate, 3));
}
function render(state) {
  current = state;
  const history = Array.isArray(state.history) ? state.history : [];
  const latest = history.length ? history[history.length-1] : null;
  const status = Object.hasOwn(STATUS,state.status) ? state.status : 'unknown';
  const cfg = state.config || {}, progress = state.progress || {};
  text('run-name', state.run || '未选择运行');
  $('status').className = `status ${status}`; text('status', STATUS[status]);
  text('pid', `PID ${state.process?.pid ?? '—'}`);
  text('updated', `更新于 ${time(new Date().toISOString(),true)}`);
  const epoch = number(progress.epoch) ?? number(latest?.epoch);
  const max = number(progress.max_epochs) ?? number(cfg.epochs);
  text('epoch-progress', `${integer(epoch)} / ${integer(max)}`);
  const step = number(progress.step), total = number(progress.total_steps);
  text('batch-progress', step !== null && total !== null ? `Batch ${integer(step)} / ${integer(total)}` : '暂无当前批次进度');
  const completed = number(latest?.epoch) ?? 0;
  bar('epoch-bar', max > 0 ? completed/max*100 : 0);
  text('epoch-summary', `已完成 ${completed} 轮${latest ? ' · 已保存验证结果' : ''}`);
  text('run-note', status === 'paused' ? '训练已暂停，硬件监测继续记录' : status === 'running' ? '训练进行中，指标在每轮验证后更新' : status === 'completed' ? '训练结束，最终测试结果已保存' : '等待训练状态更新');
  for (const metric of ['dice','iou','precision','recall']) {
    text(metric, fixed(latest?.val?.[metric]));
    const values = history.map(r=>({value:number(r.val?.[metric]),epoch:r.epoch})).filter(r=>r.value !== null);
    const best = values.reduce((a,b)=>!a || b.value>a.value ? b : a,null);
    text(`${metric}-best`, best ? `最高 ${fixed(best.value)} · 第 ${best.epoch} 轮` : '最高 —');
  }
  trainingCharts(history,cfg);
  renderResources(state); renderConfig(state,latest); renderData(state,latest);
  const logs = Array.isArray(state.latest_log) ? state.latest_log : [];
  text('log-content', logs.length ? logs.join('\n') : '暂无日志');
  text('log-status', status === 'paused' ? '已暂停 · 保留最后记录' : '随采样刷新');
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const response = await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(12000)});
    if (!response.ok) throw new Error(`服务返回 HTTP ${response.status}`);
    const state = await response.json();
    render(state);
    const warnings = Array.isArray(state.errors) ? state.errors : [];
    $('error-banner').hidden = warnings.length === 0;
    if (warnings.length) text('error-banner', `部分训练记录暂不可用：${warnings.join('；')}`);
    $('connection-dot').className = 'connection-dot ok';
    text('connection-label','本地服务已连接');
  } catch (error) {
    $('error-banner').hidden = false;
    text('error-banner',`暂时无法更新看板：${error.message}。页面保留上次记录，5 秒后重试。`);
    $('connection-dot').className = 'connection-dot offline';
    text('connection-label','连接中断，正在重试');
  } finally { refreshing = false; }
}
$('resource-mode').addEventListener('change',()=>{if(current)renderResourceChart(current);});
document.querySelectorAll('nav a').forEach(link=>link.addEventListener('click',()=>{
  document.querySelectorAll('nav a').forEach(a=>a.classList.remove('active'));
  link.classList.add('active');
}));
refresh();
setInterval(refresh,5000);
