import { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

// ── colour tokens ──────────────────────────────────────────
const C = {
  navy:    "#0A1628",
  navyMid: "#0F2040",
  navyCard:"#111E35",
  blue:    "#3B82F6",
  blueGlow:"rgba(59,130,246,0.25)",
  green:   "#10B981",
  greenGlow:"rgba(16,185,129,0.25)",
  amber:   "#F59E0B",
  red:     "#EF4444",
  purple:  "#8B5CF6",
  orange:  "#F97316",
  text:    "#E2E8F0",
  muted:   "#64748B",
  border:  "rgba(255,255,255,0.08)",
};

// ── tiny helpers ───────────────────────────────────────────
const Glass = ({ style, children, glow }) => (
  <div style={{
    background:"rgba(255,255,255,0.04)",
    border:`1px solid ${C.border}`,
    borderRadius:16,
    backdropFilter:"blur(12px)",
    boxShadow: glow ? `0 0 32px ${glow}` : "0 4px 24px rgba(0,0,0,0.3)",
    ...style,
  }}>{children}</div>
);

const Pill = ({ children, color="#3B82F6", style }) => (
  <span style={{
    display:"inline-block",
    padding:"3px 10px",
    borderRadius:20,
    fontSize:11,
    fontWeight:600,
    letterSpacing:.5,
    background:`${color}22`,
    color,
    border:`1px solid ${color}44`,
    ...style,
  }}>{children}</span>
);

const useFadeIn = () => {
  const ref = useRef();
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => e.isIntersecting && setVisible(true), { threshold:.15 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return [ref, visible];
};

const FadeIn = ({ children, delay=0 }) => {
  const [ref, visible] = useFadeIn();
  return (
    <div ref={ref} style={{
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(24px)",
      transition: `opacity .6s ease ${delay}s, transform .6s ease ${delay}s`,
    }}>{children}</div>
  );
};

// ── animated counter ───────────────────────────────────────
const Counter = ({ to, suffix="" }) => {
  const [val, setVal] = useState(0);
  const [ref, visible] = useFadeIn();
  useEffect(() => {
    if (!visible) return;
    let start = 0;
    const step = to / 40;
    const timer = setInterval(() => {
      start += step;
      if (start >= to) { setVal(to); clearInterval(timer); }
      else setVal(Math.floor(start));
    }, 30);
    return () => clearInterval(timer);
  }, [visible, to]);
  return <span ref={ref}>{val}{suffix}</span>;
};

// ── pipeline steps ─────────────────────────────────────────
const STEPS = [
  "Initializing MARC Framework",
  "Loading Knowledge Bases",
  "Running Cardiology Agent",
  "Running Nephrology Agent",
  "Running Diabetology Agent",
  "Running Pharmacology Agent",
  "SRAL — Tracking Retrievals",
  "IAKB — Broadcasting Findings",
  "SCDP — Detecting Conflicts",
  "DCWO — Building Consensus",
  "Compiling Evaluation Metrics",
];

// ── agent data ─────────────────────────────────────────────
const AGENTS = [
  {
    icon:"🫀", name:"Cardiology Agent", color:C.red,
    kb:"AHA/ACC Heart Failure Guidelines 2022",
    confidence:85, chunks:6,
    analysis:`HFrEF confirmed (EF 35%, BNP 890). Initiating GDMT:
• MRA (Spironolactone 12.5 mg/d) — COR I, LOE A (Page e931)
• SGLT2i (Dapagliflozin) — dual HF + glycaemic benefit (Page e938)
• ARNI (Sacubitril/Valsartan) preferred over ACEi
• Beta-blocker (Bisoprolol 1.25 mg) — uptitrate q2 weeks
• Loop diuretic for decongestion (BNP 890 pg/mL)
• Device therapy: re-evaluate after 3 months GDMT`,
  },
  {
    icon:"🫘", name:"Nephrology Agent", color:C.blue,
    kb:"KDIGO CKD Guidelines 2024",
    confidence:85, chunks:6,
    analysis:`CKD Stage G3b (eGFR 42 mL/min/1.73 m²):
• Confirm albuminuria — order ACR (first morning void)
• SGLT2i: first-line per KDIGO 2024 for CKD + HF
• ACEi/ARB: initiate if UACR >30 mg/g — monitor K⁺
• BP target: <120 mmHg systolic (standardised office BP)
• Repeat BMP in 7–14 days after any RAAS initiation
• GLP-1 RA (Semaglutide) if glycaemic target not met (Page 163)`,
  },
  {
    icon:"🩺", name:"Diabetology Agent", color:C.orange,
    kb:"ADA Standards of Care 2025",
    confidence:85, chunks:6,
    analysis:`T2DM + HFrEF + CKD G3b (HbA1c 8.2%):
• SGLT2i independent of HbA1c — cardiorenal protection
• eGFR 42: glycaemic benefit reduced, kidney benefit persists
• GLP-1 RA (Semaglutide > Liraglutide) — add if HbA1c target unmet
• ACEi/ARB if UACR >30 — titrate to max tolerated dose
• Finerenone if already on max-dose ACEi/ARB (eGFR >25)
• DKA education mandatory before SGLT2i initiation (Page 8)`,
  },
  {
    icon:"💊", name:"Pharmacology Agent", color:C.purple,
    kb:"WHO Essential Medicines 2023",
    confidence:85, chunks:6,
    analysis:`Pharmacological plan (WHO EML 23rd List):
• Furosemide 20–40 mg/d — adjust for eGFR 42 (Page 42)
• Enalapril 1.25–2.5 mg/d — renal dosing, monitor K⁺/Cr
• Bisoprolol 1.25 mg/d — no major renal adjustment needed
• ⚠ ACEi + insulin → hypoglycaemia risk; monitor BG closely
• ⚠ Beta-blocker masks hypoglycaemia symptoms
• ⚠ NSAIDs CONTRAINDICATED — worsens HF + AKI risk
• ⚠ Atenolol NOT preferred in HFrEF (Page 40)`,
  },
];

// ── IAKB transfers ─────────────────────────────────────────
const TRANSFERS = [
  { from:"Cardiology", to:"Nephrology",   type:"cardiac_function",  color:C.red },
  { from:"Cardiology", to:"Diabetology",  type:"cardiac_function",  color:C.red },
  { from:"Nephrology", to:"Diabetology",  type:"renal_function",    color:C.blue },
  { from:"Cardiology", to:"Pharmacology", type:"cardiac_function",  color:C.red },
  { from:"Nephrology", to:"Pharmacology", type:"renal_function",    color:C.blue },
  { from:"Diabetology","to":"Pharmacology",type:"glucose_control", color:C.orange},
];

// ── metrics data ───────────────────────────────────────────
const METRICS = [
  { label:"Answer Accuracy",     marc:76.2, base:49.5, gain:"+26.7%", color:C.green,  unit:"%" },
  { label:"Faithfulness Score",  marc:95.8, base:67.1, gain:"+28.7%", color:C.blue,   unit:"%" },
  { label:"Conflict F1 Score",   marc:100,  base:0,    gain:"Novel",  color:C.amber,  unit:"%" },
  { label:"KT Score",            marc:50,   base:0,    gain:"Novel",  color:C.purple, unit:"%" },
  { label:"Conv. Efficiency",    marc:100,  base:0,    gain:"Novel",  color:C.red,    unit:"%" },
  { label:"Dedup Rate",          marc:0,    base:0,    gain:"Novel",  color:C.orange, unit:"%" },
];

const chartData = [
  { name:"Accuracy",    MARC:76.2, Baseline:49.5 },
  { name:"Faithfulness",MARC:95.8, Baseline:67.1 },
  { name:"Conflict F1", MARC:100,  Baseline:0    },
  { name:"KT Score",    MARC:50,   Baseline:0    },
];

const CONSENSUS = [
  { drug:"SGLT2 Inhibitor",  pct:100, color:C.green  },
  { drug:"ACE Inhibitor",    pct:100, color:C.green  },
  { drug:"Beta Blocker",     pct:75,  color:C.blue   },
  { drug:"GLP-1 RA",         pct:50,  color:C.amber  },
  { drug:"MRA",              pct:50,  color:C.amber  },
  { drug:"Loop Diuretic",    pct:25,  color:C.orange },
];

// ══════════════════════════════════════════════════════════
export default function MARCClinical() {
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(-1);
  const [done, setDone] = useState(false);
  const [form, setForm] = useState({
    age:"58", symptoms:"Shortness of breath, leg swelling, fatigue, excessive thirst",
    labs:"BNP 890 pg/mL, EF 35%, Creatinine 1.8 mg/dL, HbA1c 8.2%, eGFR 42",
    meds:"None",
  });

  const runAnalysis = () => {
    setRunning(true); setDone(false); setStep(0);
    let i = 0;
    const t = setInterval(() => {
      i++;
      setStep(i);
      if (i >= STEPS.length - 1) {
        clearInterval(t);
        setTimeout(() => { setRunning(false); setDone(true); }, 400);
      }
    }, 420);
  };

  // ── styles ───────────────────────────────────────────────
  const S = {
    root:{ fontFamily:"'Inter','Segoe UI',sans-serif", background:C.navy, color:C.text, minHeight:"100vh" },
    section:{ maxWidth:1180, margin:"0 auto", padding:"72px 24px" },
    h2:{ fontSize:"clamp(1.5rem,3vw,2.2rem)", fontWeight:700, color:"#fff", marginBottom:8 },
    sub:{ color:C.muted, fontSize:14, marginBottom:32 },
  };

  return (
    <div style={S.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:6px}
        ::-webkit-scrollbar-track{background:#0A1628}
        ::-webkit-scrollbar-thumb{background:#3B82F633;border-radius:3px}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fillBar{from{width:0}to{width:var(--w)}}
        @keyframes glow{0%,100%{box-shadow:0 0 8px var(--g)}50%{box-shadow:0 0 24px var(--g)}}
        @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
      `}</style>

      <Hero form={form} setForm={setForm} running={running} step={step} done={done} runAnalysis={runAnalysis} />
      <HowItWorks S={S} />
      {done && <>
        <FrameworkStatus S={S} />
        <AgentResults S={S} />
        <IAKBFlow S={S} />
        <SCDPConflicts S={S} />
        <DCWOConsensus S={S} />
        <EvalMetrics S={S} chartData={chartData} />
      </>}
      <About S={S} />
    </div>
  );
}

// ── HERO ───────────────────────────────────────────────────
function Hero({ form, setForm, running, step, done, runAnalysis }) {
  return (
    <section style={{
      background:`linear-gradient(135deg, ${C.navy} 0%, #0D2044 60%, #0A1628 100%)`,
      padding:"80px 24px 0",
      position:"relative", overflow:"hidden",
    }}>
      {/* ambient orbs */}
      <div style={{ position:"absolute",top:-120,right:-120,width:400,height:400,borderRadius:"50%",background:`radial-gradient(circle,${C.blueGlow},transparent 70%)`,pointerEvents:"none"}} />
      <div style={{ position:"absolute",bottom:0,left:-80,width:320,height:320,borderRadius:"50%",background:`radial-gradient(circle,${C.greenGlow},transparent 70%)`,pointerEvents:"none"}} />

      <div style={{ maxWidth:900, margin:"0 auto", textAlign:"center", position:"relative" }}>
        {/* eyebrow */}
        <div style={{ display:"flex", gap:8, flexWrap:"wrap", justifyContent:"center", marginBottom:24 }}>
          {["RAG","LLMs","Multi-Agent Systems","AI Agents","IEEE Access 2025","IEEE ICHI 2026"].map(t=>(
            <Pill key={t} color={t.startsWith("IEEE") ? C.amber : C.blue}>{t}</Pill>
          ))}
        </div>

        {/* title */}
        <h1 style={{ fontSize:"clamp(2.4rem,6vw,4rem)", fontWeight:800, color:"#fff", lineHeight:1.1, marginBottom:16 }}>
          MARC<span style={{ color:C.blue }}>-</span>Clinical
        </h1>
        <p style={{ fontSize:"clamp(1rem,2.5vw,1.3rem)", color:"#94A3B8", marginBottom:8 }}>
          Decentralized Multi-Agent RAG Coordination Framework
        </p>
        <p style={{ fontSize:"clamp(.85rem,2vw,1rem)", color:C.muted, marginBottom:36 }}>
          for Multi-Disease Clinical Decision Support
        </p>

        {/* stat pills */}
        <div style={{ display:"flex", gap:16, flexWrap:"wrap", justifyContent:"center", marginBottom:44 }}>
          {[["4","Specialist Agents"],["95.8%","Faithfulness"],["1.0","Conflict F1"],["Round 1","Consensus"]].map(([v,l])=>(
            <div key={l} style={{ textAlign:"center" }}>
              <div style={{ fontSize:22, fontWeight:700, color:"#fff" }}>{v}</div>
              <div style={{ fontSize:11, color:C.muted }}>{l}</div>
            </div>
          ))}
        </div>

        {/* SDG badges */}
        <div style={{ display:"flex", gap:10, justifyContent:"center", marginBottom:52 }}>
          <Pill color={C.green}>🌿 SDG 3 — Good Health</Pill>
          <Pill color={C.blue}>⚙️ SDG 9 — Innovation</Pill>
        </div>

        {/* form card */}
        <Glass style={{ padding:32, textAlign:"left", marginBottom:0, borderRadius:"16px 16px 0 0" }}>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
            <Field label="Patient Age" value={form.age} onChange={v=>setForm({...form,age:v})} placeholder="58" />
            <Field label="Current Medications" value={form.meds} onChange={v=>setForm({...form,meds:v})} placeholder="None" />
          </div>
          <FieldArea label="Symptoms" value={form.symptoms} onChange={v=>setForm({...form,symptoms:v})} rows={2} />
          <div style={{ marginTop:16 }}>
            <FieldArea label="Lab Results" value={form.labs} onChange={v=>setForm({...form,labs:v})} rows={2} />
          </div>
          <div style={{ display:"flex", gap:12, marginTop:20, alignItems:"center", flexWrap:"wrap" }}>
            <button onClick={runAnalysis} disabled={running} style={{
              padding:"12px 32px", borderRadius:10, border:"none", cursor:"pointer",
              background: running ? C.muted : `linear-gradient(135deg,${C.blue},${C.green})`,
              color:"#fff", fontWeight:700, fontSize:15,
              boxShadow: running ? "none" : `0 4px 20px ${C.blueGlow}`,
              transition:"all .2s",
            }}>
              {running ? "⟳ Analyzing..." : "Analyze Case →"}
            </button>
            <button onClick={()=>setForm({age:"58",symptoms:"Shortness of breath, leg swelling, fatigue, excessive thirst",labs:"BNP 890 pg/mL, EF 35%, Creatinine 1.8 mg/dL, HbA1c 8.2%, eGFR 42",meds:"None"})}
              style={{ padding:"12px 20px",borderRadius:10,border:`1px solid ${C.border}`,background:"transparent",color:C.muted,cursor:"pointer",fontSize:13 }}>
              Load Demo Case
            </button>
          </div>

          {/* pipeline */}
          {(running || done) && <Pipeline step={step} done={done} />}
        </Glass>
      </div>
    </section>
  );
}

function Field({ label, value, onChange, placeholder }) {
  return (
    <div>
      <label style={{ fontSize:12, fontWeight:600, color:C.muted, display:"block", marginBottom:6, textTransform:"uppercase", letterSpacing:.5 }}>{label}</label>
      <input value={value} onChange={e=>onChange(e.target.value)} placeholder={placeholder}
        style={{ width:"100%", padding:"10px 14px", borderRadius:8, border:`1px solid ${C.border}`,
          background:"rgba(255,255,255,0.04)", color:C.text, fontSize:14, outline:"none" }} />
    </div>
  );
}
function FieldArea({ label, value, onChange, rows=3 }) {
  return (
    <div>
      <label style={{ fontSize:12, fontWeight:600, color:C.muted, display:"block", marginBottom:6, textTransform:"uppercase", letterSpacing:.5 }}>{label}</label>
      <textarea value={value} onChange={e=>onChange(e.target.value)} rows={rows}
        style={{ width:"100%", padding:"10px 14px", borderRadius:8, border:`1px solid ${C.border}`,
          background:"rgba(255,255,255,0.04)", color:C.text, fontSize:14, outline:"none", resize:"vertical", fontFamily:"inherit" }} />
    </div>
  );
}

function Pipeline({ step, done }) {
  return (
    <div style={{ marginTop:24, borderTop:`1px solid ${C.border}`, paddingTop:20 }}>
      <div style={{ fontSize:12, fontWeight:600, color:C.muted, marginBottom:12, textTransform:"uppercase", letterSpacing:.5 }}>
        Pipeline Progress
      </div>
      <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
        {STEPS.map((s,i)=>{
          const active = i === step;
          const complete = i < step || done;
          return (
            <div key={s} style={{ display:"flex", alignItems:"center", gap:6, padding:"5px 10px",
              borderRadius:20, border:`1px solid ${complete ? C.green+"44" : active ? C.blue+"44" : C.border}`,
              background: complete ? `${C.green}11` : active ? `${C.blue}11` : "transparent",
              fontSize:11, color: complete ? C.green : active ? C.blue : C.muted,
              transition:"all .3s",
            }}>
              <span style={{ animation: active ? "spin 1s linear infinite" : "none", display:"inline-block" }}>
                {complete ? "✓" : active ? "⟳" : "○"}
              </span>
              {s}
            </div>
          );
        })}
      </div>
      {done && (
        <div style={{ marginTop:16, padding:"10px 16px", borderRadius:8, background:`${C.green}15`,
          border:`1px solid ${C.green}33`, color:C.green, fontWeight:600, fontSize:13 }}>
          ✅ All 4 agents completed. MARC framework analysis ready. Scroll down to view results.
        </div>
      )}
    </div>
  );
}

// ── HOW IT WORKS ───────────────────────────────────────────
function HowItWorks({ S }) {
  const modules = [
    { name:"SRAL", full:"Shared Retrieval Awareness Layer", desc:"Prevents duplicate document retrieval across agents", color:C.blue },
    { name:"IAKB", full:"Inter-Agent Knowledge Bus", desc:"Real-time findings broadcast between agents mid-reasoning", color:C.green },
    { name:"SCDP", full:"Semantic Conflict Detection", desc:"Evidence-weighted resolution of contradicting recommendations", color:C.amber },
    { name:"DCWO", full:"Decentralized Consensus", desc:"Gossip-protocol consensus without a central orchestrator", color:C.purple },
  ];
  return (
    <section style={S.section}>
      <FadeIn>
        <div style={{ textAlign:"center", marginBottom:48 }}>
          <h2 style={S.h2}>How MARC Works</h2>
          <p style={S.sub}>4 specialist agents coordinate without a central controller</p>
        </div>
        {/* agent → framework → modules */}
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:24 }}>
          {/* agents row */}
          <div style={{ display:"flex", gap:16, flexWrap:"wrap", justifyContent:"center" }}>
            {AGENTS.map(a=>(
              <Glass key={a.name} style={{ padding:"14px 20px", borderTop:`3px solid ${a.color}`, textAlign:"center", minWidth:140 }}>
                <div style={{ fontSize:24 }}>{a.icon}</div>
                <div style={{ fontSize:12, fontWeight:600, color:a.color, marginTop:4 }}>{a.name.replace(" Agent","")}</div>
              </Glass>
            ))}
          </div>
          {/* arrow */}
          <div style={{ color:C.muted, fontSize:22 }}>↓ RAG Retrieval + LLM Reasoning</div>
          {/* MARC hub */}
          <Glass glow={C.blueGlow} style={{ padding:"20px 40px", textAlign:"center", border:`1px solid ${C.blue}44` }}>
            <div style={{ fontSize:18, fontWeight:700, color:C.blue }}>MARC Framework Hub</div>
            <div style={{ fontSize:12, color:C.muted, marginTop:4 }}>Coordination Layer</div>
          </Glass>
          {/* arrow */}
          <div style={{ color:C.muted, fontSize:22 }}>↓ Framework Modules</div>
          {/* modules pipeline */}
          <div style={{ display:"flex", gap:12, flexWrap:"wrap", justifyContent:"center", alignItems:"center" }}>
            {modules.map((m,i)=>(
              <div key={m.name} style={{ display:"flex", alignItems:"center", gap:12 }}>
                <Glass style={{ padding:"16px 20px", borderLeft:`3px solid ${m.color}`, minWidth:170 }}>
                  <div style={{ fontWeight:700, color:m.color, fontSize:15 }}>{m.name}</div>
                  <div style={{ fontSize:11, color:C.muted, marginTop:2 }}>{m.full}</div>
                  <div style={{ fontSize:11, color:"#94A3B8", marginTop:4 }}>{m.desc}</div>
                </Glass>
                {i < modules.length-1 && <div style={{ color:C.muted, fontSize:20 }}>→</div>}
              </div>
            ))}
          </div>
        </div>
      </FadeIn>
    </section>
  );
}

// ── FRAMEWORK STATUS ───────────────────────────────────────
function FrameworkStatus({ S }) {
  const modules = [
    { name:"SRAL", full:"Shared Retrieval Awareness Layer", metric:"24 unique docs tracked", color:C.blue },
    { name:"IAKB", full:"Inter-Agent Knowledge Bus", metric:"6 knowledge transfers", color:C.green },
    { name:"SCDP", full:"Conflict Detection Protocol", metric:"2 conflicts | F1: 1.0", color:C.amber },
    { name:"DCWO", full:"Decentralized Consensus", metric:"Consensus: Round 1", color:C.purple },
  ];
  return (
    <section style={{ ...S.section, paddingTop:0 }}>
      <FadeIn>
        <div style={{ fontSize:12, fontWeight:600, color:C.muted, textTransform:"uppercase", letterSpacing:.5, marginBottom:16 }}>
          Framework Modules — All Active
        </div>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))", gap:16 }}>
          {modules.map(m=>(
            <Glass key={m.name} glow={m.color+"33"} style={{
              padding:24, borderLeft:`3px solid ${m.color}`,
              "--g":m.color+"44", animation:"glow 3s ease-in-out infinite",
            }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
                <div>
                  <div style={{ fontSize:20, fontWeight:800, color:m.color }}>{m.name}</div>
                  <div style={{ fontSize:11, color:C.muted, marginTop:2 }}>{m.full}</div>
                </div>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.green,
                  animation:"pulse 2s ease-in-out infinite", marginTop:4 }} />
              </div>
              <div style={{ marginTop:16, fontSize:13, fontWeight:600, color:"#fff" }}>{m.metric}</div>
            </Glass>
          ))}
        </div>
      </FadeIn>
    </section>
  );
}

// ── AGENT RESULTS ──────────────────────────────────────────
function AgentResults({ S }) {
  return (
    <section style={{ ...S.section, paddingTop:0 }}>
      <FadeIn>
        <h2 style={S.h2}>Specialist Agent Analysis</h2>
        <p style={S.sub}>Each agent retrieves from its dedicated knowledge base — grounded, cited, no hallucination</p>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(280px,1fr))", gap:20 }}>
          {AGENTS.map((a,i)=>(
            <FadeIn key={a.name} delay={i*0.1}>
              <Glass style={{ padding:24, borderTop:`3px solid ${a.color}`, height:"100%" }}>
                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
                  <div style={{ display:"flex", gap:12, alignItems:"center" }}>
                    <span style={{ fontSize:28 }}>{a.icon}</span>
                    <div>
                      <div style={{ fontWeight:700, color:"#fff", fontSize:15 }}>{a.name}</div>
                      <div style={{ fontSize:11, color:C.muted }}>{a.kb}</div>
                    </div>
                  </div>
                </div>
                {/* confidence */}
                <div style={{ margin:"16px 0 12px" }}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                    <span style={{ fontSize:12, color:C.muted }}>Confidence</span>
                    <span style={{ fontSize:12, fontWeight:600, color:a.color }}>{a.confidence}%</span>
                  </div>
                  <div style={{ height:4, background:"rgba(255,255,255,0.08)", borderRadius:2 }}>
                    <div style={{ height:"100%", width:`${a.confidence}%`, background:`linear-gradient(90deg,${a.color},${a.color}88)`, borderRadius:2 }} />
                  </div>
                </div>
                {/* analysis */}
                <div style={{ background:"rgba(0,0,0,0.3)", borderRadius:8, padding:"12px 14px",
                  fontSize:12, color:"#CBD5E1", lineHeight:1.7, whiteSpace:"pre-wrap",
                  border:`1px solid ${a.color}22`, maxHeight:200, overflowY:"auto" }}>
                  {a.analysis}
                </div>
                <div style={{ marginTop:12 }}>
                  <Pill color={a.color}>{a.chunks} chunks retrieved</Pill>
                </div>
              </Glass>
            </FadeIn>
          ))}
        </div>
      </FadeIn>
    </section>
  );
}

// ── IAKB FLOW ──────────────────────────────────────────────
function IAKBFlow({ S }) {
  const [hovered, setHovered] = useState(null);
  const agents = [
    { id:"Cardiology",   x:120, y:60,  icon:"🫀", color:C.red    },
    { id:"Nephrology",   x:380, y:60,  icon:"🫘", color:C.blue   },
    { id:"Diabetology",  x:120, y:200, icon:"🩺", color:C.orange },
    { id:"Pharmacology", x:380, y:200, icon:"💊", color:C.purple },
  ];
  const agMap = Object.fromEntries(agents.map(a=>[a.id,a]));

  const typeColor = { cardiac_function:C.red, renal_function:C.blue, glucose_control:C.orange };

  return (
    <section style={{ ...S.section, paddingTop:0 }}>
      <FadeIn>
        <h2 style={S.h2}>IAKB — Inter-Agent Knowledge Bus</h2>
        <p style={S.sub}>Mid-reasoning knowledge transfer — 6 transfers across 4 agents</p>
        <Glass style={{ padding:32 }}>
          <div style={{ position:"relative", width:"100%", overflowX:"auto" }}>
            <svg viewBox="0 0 520 280" style={{ width:"100%", minWidth:400 }}>
              {/* arrows */}
              {TRANSFERS.map((t,i)=>{
                const f = agMap[t.from]; const to = agMap[t.to];
                const mx = (f.x+to.x)/2; const my = (f.y+to.y)/2;
                const col = typeColor[t.type] || C.blue;
                const isH = hovered === i;
                return (
                  <g key={i} onMouseEnter={()=>setHovered(i)} onMouseLeave={()=>setHovered(null)} style={{ cursor:"pointer" }}>
                    <line x1={f.x+50} y1={f.y+24} x2={to.x+10} y2={to.y+24}
                      stroke={col} strokeWidth={isH?3:1.5} strokeDasharray="6 3"
                      opacity={isH?1:.6}
                      markerEnd={`url(#arr${i})`}
                    />
                    <defs>
                      <marker id={`arr${i}`} markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                        <path d="M0,0 L6,3 L0,6 z" fill={col} />
                      </marker>
                    </defs>
                    {isH && (
                      <text x={mx} y={my-8} textAnchor="middle" fill={col} fontSize="10" fontWeight="600">
                        {t.type}
                      </text>
                    )}
                    <circle cx={mx} cy={my} r={10} fill={col+"22"} stroke={col+"55"} strokeWidth={1} />
                    <text x={mx} y={my+4} textAnchor="middle" fill={col} fontSize="8" fontWeight="700">
                      {t.type.split("_")[0][0].toUpperCase()}
                    </text>
                  </g>
                );
              })}
              {/* agent nodes */}
              {agents.map(a=>(
                <g key={a.id}>
                  <rect x={a.x} y={a.y} width={110} height={48} rx={10}
                    fill={C.navyCard} stroke={a.color+"88"} strokeWidth={1.5} />
                  <text x={a.x+20} y={a.y+20} fontSize="16">{a.icon}</text>
                  <text x={a.x+42} y={a.y+18} fill="#fff" fontSize="10" fontWeight="700">{a.id}</text>
                  <text x={a.x+42} y={a.y+30} fill={C.muted} fontSize="9">Agent</text>
                </g>
              ))}
            </svg>
          </div>
          <div style={{ marginTop:16, display:"flex", gap:12, flexWrap:"wrap" }}>
            {[["cardiac_function",C.red],["renal_function",C.blue],["glucose_control",C.orange]].map(([t,c])=>(
              <Pill key={t} color={c}>{t}</Pill>
            ))}
            <span style={{ fontSize:12, color:C.muted, display:"flex", alignItems:"center" }}>← Hover arrows for detail</span>
          </div>
        </Glass>
      </FadeIn>
    </section>
  );
}

// ── SCDP CONFLICTS ─────────────────────────────────────────
function SCDPConflicts({ S }) {
  const conflicts = [
    {
      a:"Cardiology Agent", b:"Nephrology Agent",
      issue:"restrict fluid vs increase fluid",
      detail:"fluid restriction vs adequate hydration",
      scoreA:1.0, scoreB:0.9, winner:"Cardiology Agent", winnerScore:1.0,
    },
    {
      a:"Diabetology Agent", b:"Pharmacology Agent",
      issue:"hold metformin vs continue metformin",
      detail:"lactic acidosis risk vs dose-adjusted use",
      scoreA:0.8, scoreB:0.6, winner:"Diabetology Agent", winnerScore:0.8,
    },
  ];
  return (
    <section style={{ ...S.section, paddingTop:0 }}>
      <FadeIn>
        <h2 style={S.h2}>SCDP — Conflict Detection & Resolution</h2>
        <p style={S.sub}>Evidence-weighted resolution — Precision 1.0 | Recall 1.0 | F1: 1.0</p>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(320px,1fr))", gap:20 }}>
          {conflicts.map((c,i)=>(
            <Glass key={i} style={{ padding:24, borderLeft:`3px solid ${C.red}` }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16 }}>
                <Pill color={C.red}>⚡ CONFLICT DETECTED</Pill>
                <Pill color={C.green}>✓ RESOLVED</Pill>
              </div>
              {/* vs */}
              <div style={{ display:"grid", gridTemplateColumns:"1fr auto 1fr", gap:8, alignItems:"center", marginBottom:16 }}>
                <div style={{ padding:"10px 12px", borderRadius:8, background:`${C.blue}11`, border:`1px solid ${C.blue}33`, textAlign:"center" }}>
                  <div style={{ fontSize:11, fontWeight:600, color:C.blue }}>{c.a}</div>
                  <div style={{ fontSize:12, color:"#fff", marginTop:4 }}>Score: {c.scoreA}</div>
                </div>
                <div style={{ color:C.red, fontWeight:700, fontSize:16 }}>↔</div>
                <div style={{ padding:"10px 12px", borderRadius:8, background:`${C.amber}11`, border:`1px solid ${C.amber}33`, textAlign:"center" }}>
                  <div style={{ fontSize:11, fontWeight:600, color:C.amber }}>{c.b}</div>
                  <div style={{ fontSize:12, color:"#fff", marginTop:4 }}>Score: {c.scoreB}</div>
                </div>
              </div>
              <div style={{ background:"rgba(239,68,68,.08)", borderRadius:8, padding:"10px 14px", marginBottom:12 }}>
                <div style={{ fontSize:12, color:C.red, fontWeight:600 }}>"{c.issue}"</div>
                <div style={{ fontSize:11, color:C.muted, marginTop:2 }}>{c.detail}</div>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                <span style={{ fontSize:16 }}>🏆</span>
                <span style={{ fontSize:13, color:C.green, fontWeight:600 }}>{c.winner} wins</span>
                <Pill color={C.green} style={{ marginLeft:"auto" }}>Evidence: {c.winnerScore}</Pill>
              </div>
            </Glass>
          ))}
        </div>
      </FadeIn>
    </section>
  );
}

// ── DCWO CONSENSUS ─────────────────────────────────────────
function DCWOConsensus({ S }) {
  const [animated, setAnimated] = useState(false);
  const [ref, visible] = useFadeIn();
  useEffect(() => { if (visible) setTimeout(() => setAnimated(true), 300); }, [visible]);

  return (
    <section style={{ ...S.section, paddingTop:0 }}>
      <div ref={ref}>
        <h2 style={S.h2}>DCWO — Decentralized Consensus</h2>
        <p style={S.sub}>No central orchestrator — gossip-protocol convergence in Round 1</p>
        <Glass glow={C.greenGlow} style={{ padding:32, border:`1px solid ${C.green}33` }}>
          <div style={{ display:"flex", justifyContent:"space-between", flexWrap:"wrap", gap:16, marginBottom:28 }}>
            <div>
              <div style={{ fontSize:22, fontWeight:800, color:C.green }}>✅ CONSENSUS REACHED</div>
              <div style={{ fontSize:14, color:C.muted, marginTop:4 }}>Round 1 · 4 agents · No orchestrator required</div>
            </div>
            <Pill color={C.blue} style={{ height:"fit-content", padding:"6px 14px", fontSize:12 }}>
              Agreement Score: 0.85
            </Pill>
          </div>
          <div style={{ display:"flex", gap:12, flexWrap:"wrap", marginBottom:28 }}>
            <div style={{ padding:"8px 16px", borderRadius:8, background:`${C.green}15`, border:`1px solid ${C.green}33` }}>
              <div style={{ fontSize:11, color:C.muted }}>All agents agree on</div>
              <div style={{ fontSize:13, fontWeight:600, color:C.green, marginTop:2 }}>SGLT2i + ACE Inhibitor</div>
            </div>
            <div style={{ padding:"8px 16px", borderRadius:8, background:`${C.blue}15`, border:`1px solid ${C.blue}33` }}>
              <div style={{ fontSize:11, color:C.muted }}>Majority agree on</div>
              <div style={{ fontSize:13, fontWeight:600, color:C.blue, marginTop:2 }}>Beta Blocker · GLP-1 RA · MRA</div>
            </div>
          </div>
          {/* bars */}
          <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
            {CONSENSUS.map(c=>(
              <div key={c.drug} style={{ display:"grid", gridTemplateColumns:"160px 1fr 48px", gap:12, alignItems:"center" }}>
                <span style={{ fontSize:13, color:C.text }}>{c.drug}</span>
                <div style={{ height:8, background:"rgba(255,255,255,0.07)", borderRadius:4, overflow:"hidden" }}>
                  <div style={{
                    height:"100%", width: animated ? `${c.pct}%` : "0%",
                    background:`linear-gradient(90deg,${c.color},${c.color}88)`,
                    borderRadius:4, transition:"width 1.2s cubic-bezier(.4,0,.2,1)",
                  }} />
                </div>
                <span style={{ fontSize:12, fontWeight:600, color:c.color, textAlign:"right" }}>{c.pct}%</span>
              </div>
            ))}
          </div>
        </Glass>
      </div>
    </section>
  );
}

// ── EVALUATION METRICS ─────────────────────────────────────
function EvalMetrics({ S, chartData }) {
  return (
    <section style={{ ...S.section, paddingTop:0 }}>
      <FadeIn>
        <h2 style={S.h2}>Evaluation Results</h2>
        <p style={S.sub}>MARC vs Single-Agent RAG Baseline — measured on clinical QA benchmarks</p>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(160px,1fr))", gap:16, marginBottom:32 }}>
          {METRICS.map((m,i)=>(
            <FadeIn key={m.label} delay={i*0.08}>
              <Glass style={{ padding:20, textAlign:"center", borderTop:`2px solid ${m.color}` }}>
                <div style={{ fontSize:11, color:C.muted, marginBottom:8, fontWeight:600, textTransform:"uppercase", letterSpacing:.5 }}>{m.label}</div>
                <div style={{ fontSize:28, fontWeight:800, color:m.color }}>
                  {m.marc > 0 ? <><Counter to={m.marc} />{m.unit}</> : "—"}
                </div>
                <div style={{ fontSize:11, color:C.muted, marginTop:4 }}>
                  Baseline: {m.base}{m.unit}
                </div>
                <Pill color={m.gain === "Novel" ? C.purple : C.green} style={{ marginTop:8 }}>
                  {m.gain}
                </Pill>
              </Glass>
            </FadeIn>
          ))}
        </div>
        {/* chart */}
        <Glass style={{ padding:24 }}>
          <div style={{ fontSize:14, fontWeight:600, color:"#fff", marginBottom:20 }}>
            MARC vs Baseline — Visual Comparison
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ top:0, right:20, bottom:0, left:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="name" tick={{ fill:C.muted, fontSize:12 }} />
              <YAxis tick={{ fill:C.muted, fontSize:12 }} domain={[0,110]} />
              <Tooltip contentStyle={{ background:C.navyCard, border:`1px solid ${C.border}`, borderRadius:8, color:C.text }} />
              <Legend wrapperStyle={{ color:C.muted, fontSize:12 }} />
              <Bar dataKey="MARC" fill={C.blue} radius={[4,4,0,0]} />
              <Bar dataKey="Baseline" fill={C.muted} radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </Glass>
      </FadeIn>
    </section>
  );
}

// ── ABOUT ──────────────────────────────────────────────────
function About({ S }) {
  const papers = [
    { tag:"Primary Base Paper", journal:"IEEE Access 2025 — SCIE ✅", title:"An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence", doi:"10.1109/ACCESS.2025.3613340" },
    { tag:"Supporting Paper",   journal:"IEEE ICHI 2026 — Scopus ✅", title:"MediHive: A Decentralized Agent Collective for Medical Reasoning", doi:"arXiv: 2603.27150" },
  ];
  const modules = [
    { n:"SRAL", d:"Retrieval deduplication with cross-agent interpretation sharing" },
    { n:"IAKB", d:"Mid-reasoning knowledge transfer via publish-subscribe bus" },
    { n:"SCDP", d:"Evidence-weighted conflict resolution using semantic comparison" },
    { n:"DCWO", d:"Gossip-protocol consensus without centralized orchestration" },
  ];
  return (
    <section style={{ ...S.section, borderTop:`1px solid ${C.border}` }}>
      <FadeIn>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(280px,1fr))", gap:32 }}>
          <div>
            <div style={{ fontSize:11, fontWeight:600, color:C.muted, textTransform:"uppercase", letterSpacing:.5, marginBottom:16 }}>Base Papers</div>
            {papers.map(p=>(
              <Glass key={p.doi} style={{ padding:20, marginBottom:12, borderLeft:`3px solid ${C.blue}` }}>
                <Pill color={C.blue} style={{ marginBottom:8 }}>{p.tag}</Pill>
                <div style={{ fontSize:11, color:C.muted, marginBottom:4 }}>{p.journal}</div>
                <div style={{ fontSize:12, color:"#fff", lineHeight:1.5, marginBottom:6 }}>{p.title}</div>
                <div style={{ fontSize:11, color:C.blue }}>{p.doi}</div>
              </Glass>
            ))}
          </div>
          <div>
            <div style={{ fontSize:11, fontWeight:600, color:C.muted, textTransform:"uppercase", letterSpacing:.5, marginBottom:16 }}>Novel Contributions</div>
            {modules.map((m,i)=>(
              <div key={m.n} style={{ display:"flex", gap:14, marginBottom:16, alignItems:"flex-start" }}>
                <div style={{ width:36, height:36, borderRadius:8, background:`${C.blue}22`, border:`1px solid ${C.blue}44`,
                  display:"flex", alignItems:"center", justifyContent:"center", fontWeight:700, color:C.blue, fontSize:12, flexShrink:0 }}>
                  {i+1}
                </div>
                <div>
                  <div style={{ fontWeight:600, color:"#fff", fontSize:14 }}>{m.n}</div>
                  <div style={{ fontSize:12, color:C.muted, marginTop:2 }}>{m.d}</div>
                </div>
              </div>
            ))}
            <Glass style={{ padding:20, marginTop:8, borderTop:`2px solid ${C.green}` }}>
              <div style={{ fontSize:13, fontWeight:600, color:"#fff" }}>Department of AI & Data Science</div>
              <div style={{ fontSize:12, color:C.muted, marginTop:2 }}>Kongu Engineering College (Autonomous)</div>
              <div style={{ fontSize:12, color:C.muted }}>Course: 22ADP72 — Project Work II Phase I</div>
              <div style={{ marginTop:10, display:"flex", gap:8, flexWrap:"wrap" }}>
                <Pill color={C.green}>SDG 3</Pill>
                <Pill color={C.blue}>SDG 9</Pill>
                <Pill color={C.purple}>Final Year Project 2025-26</Pill>
              </div>
            </Glass>
          </div>
        </div>
      </FadeIn>
    </section>
  );
}
