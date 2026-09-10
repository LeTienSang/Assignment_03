import { useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  ChartNoAxesCombined,
  CircleAlert,
  LoaderCircle,
  RotateCcw,
  ShoppingBag,
  Target,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type FormState = {
  views: number;
  cart_additions: number;
  total_spent: number;
  days_since_last_active: number;
};

type Prediction = {
  purchase_probability: number;
  segment: string;
  churn_risk: string;
};

const initialForm: FormState = {
  views: 14,
  cart_additions: 3,
  total_spent: 185,
  days_since_last_active: 6,
};

const funnelData = [
  { label: "Lượt xem sản phẩm", value: 86, color: "#8ce3c3" },
  { label: "Thêm vào giỏ", value: 54, color: "#70b7ff" },
  { label: "Bắt đầu thanh toán", value: 38, color: "#b5a2ff" },
  { label: "Mua hàng", value: 24, color: "#f5c96a" },
];

function App() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (field: keyof FormState, value: string) => {
    setForm((current) => ({ ...current, [field]: Number(value) }));
  };

  const predict = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("http://127.0.0.1:8000/predict/ecommerce-behavior", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) throw new Error("Prediction request failed");
      setPrediction((await response.json()) as Prediction);
    } catch {
      setError("Could not connect to the prediction API. Start FastAPI on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setForm(initialForm);
    setPrediction(null);
    setError("");
  };

  const probability = prediction ? Math.round(prediction.purchase_probability * 100) : 0;
  const riskTone = prediction?.churn_risk === "High" ? "risk-high" : prediction?.churn_risk === "Medium" ? "risk-medium" : "risk-low";
  const segmentLabel = prediction ? translateSegment(prediction.segment) : "Chưa có dữ liệu";
  const riskLabel = prediction ? translateRisk(prediction.churn_risk) : "Chưa có dữ liệu";

  return (
    <main className="min-h-screen overflow-hidden bg-ink text-slate-100">
      <div className="mx-auto max-w-[1440px] px-5 py-6 sm:px-8 lg:px-12 lg:py-10">
        <header className="mb-10 flex items-center justify-between border-b border-white/10 pb-6">
          <div className="flex items-center gap-3">
            <div className="brand-mark"><Activity size={20} strokeWidth={2.5} /></div>
            <div>
              <p className="font-display text-lg font-bold tracking-tight">PULSEBOARD</p>
              <p className="eyebrow">Customer behavior / 02</p>
            </div>
          </div>
          <div className="hidden items-center gap-2 text-xs text-slate-400 sm:flex">
            <span className="live-dot" /> MÔ HÌNH ĐANG HOẠT ĐỘNG
            <span className="mx-2 text-white/20">|</span>
            <span>BỘ PHÂN LOẠI RF · v1.0</span>
          </div>
        </header>

        <section className="mb-10 max-w-3xl animate-rise">
          <p className="eyebrow mb-3 text-mint">Customer behavior dashboard</p>
          <h1 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl">Khả năng mua hàng</h1>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-slate-400">Nhập hoạt động của khách hàng để ước tính khả năng mua và nguy cơ rời bỏ.</p>
        </section>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,0.88fr)_minmax(0,1.12fr)]">
          <section className="panel animate-rise delay-one">
            <div className="mb-7 flex items-start justify-between">
              <div>
                <p className="eyebrow mb-2 text-slate-500">01 / Dữ liệu đầu vào</p>
                <h2 className="font-display text-2xl font-bold text-white">Thông tin khách hàng</h2>
              </div>
              <div className="icon-tile"><BrainCircuit size={20} /></div>
            </div>
            <form onSubmit={predict} className="space-y-5">
              <Field label="Lượt xem sản phẩm" value={form.views} suffix="lượt" onChange={(value) => updateField("views", value)} />
              <Field label="Số lần thêm vào giỏ" value={form.cart_additions} suffix="lần" onChange={(value) => updateField("cart_additions", value)} />
              <Field label="Tổng chi tiêu" value={form.total_spent} suffix="USD" onChange={(value) => updateField("total_spent", value)} />
              <Field label="Số ngày không hoạt động" value={form.days_since_last_active} suffix="ngày" onChange={(value) => updateField("days_since_last_active", value)} />
              <div className="flex gap-3 pt-3">
                <button className="primary-button" type="submit" disabled={loading}>{loading ? <LoaderCircle className="animate-spin" size={18} /> : <Target size={18} />}{loading ? "Đang phân tích" : "Dự đoán"}<ArrowUpRight size={16} /></button>
                <button className="reset-button" type="button" onClick={reset} aria-label="Reset form" title="Reset form"><RotateCcw size={18} /></button>
              </div>
            </form>
            {error && <div className="mt-5 flex items-start gap-2 rounded-lg border border-red-400/20 bg-red-400/10 p-3 text-sm text-red-200"><CircleAlert size={17} className="mt-0.5 shrink-0" />Không thể kết nối đến API dự đoán. Hãy khởi động FastAPI trên cổng 8000.</div>}
          </section>

          <section className="space-y-6 animate-rise delay-two">
            <div className="result-panel">
              <div className="flex items-start justify-between">
                <div><p className="eyebrow mb-2 text-slate-500">02 / Kết quả mô hình</p><h2 className="font-display text-2xl font-bold text-white">Khả năng mua hàng</h2></div>
                <div className="icon-tile mint-tile"><ChartNoAxesCombined size={20} /></div>
              </div>
              <div className="mt-8 flex flex-wrap items-end gap-x-6 gap-y-5">
                <div className="probability"><span>{probability}</span><small>%</small></div>
                <div className="pb-2"><p className="mb-2 text-sm text-slate-400">Xác suất mua hàng</p><div className="progress-track"><div className="progress-value" style={{ width: `${probability}%` }} /></div></div>
              </div>
              <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3">
                <Metric icon={<Users size={16} />} label="Phân nhóm" value={segmentLabel} />
                <Metric icon={<CircleAlert size={16} />} label="Nguy cơ rời bỏ" value={riskLabel} tone={prediction ? riskTone : ""} />
                <Metric icon={<ShoppingBag size={16} />} label="Mô hình" value="Random Forest" />
              </div>
            </div>

            <div className="panel chart-panel">
              <div className="mb-5 flex items-center justify-between"><div><p className="eyebrow mb-2 text-slate-500">03 / Tổng quan</p><h2 className="font-display text-2xl font-bold text-white">Phân tích phễu chuyển đổi</h2></div><span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-400">30 ngày qua</span></div>
              <div className="h-[245px] w-full"><ResponsiveContainer width="100%" height="100%"><BarChart data={funnelData} layout="vertical" margin={{ top: 4, right: 12, left: 8, bottom: 4 }} barCategoryGap="22%"><CartesianGrid horizontal={false} stroke="#dce5e2" /><XAxis type="number" domain={[0, 100]} hide /><YAxis type="category" dataKey="label" axisLine={false} tickLine={false} width={130} tick={{ fill: "#64716f", fontSize: 12 }} /><Tooltip cursor={{ fill: "#eef8f4" }} contentStyle={{ background: "#ffffff", border: "1px solid #dce5e2", borderRadius: 8, color: "#1f2c29" }} formatter={(value) => [`${value}%`, "Tỷ lệ"]} /><Bar dataKey="value" radius={[0, 4, 4, 0]} fill="#51b892" /></BarChart></ResponsiveContainer></div>
              <div className="mt-2 flex items-center gap-2 text-xs text-slate-500"><span className="h-2 w-2 rounded-full bg-mint" /> Tỷ lệ chuyển đổi theo hành động</div>
            </div>
          </section>
        </div>
        <footer className="mt-8 flex items-center justify-between border-t border-white/10 pt-5 text-xs text-slate-600"><span>DỮ LIỆU / PHÂN TÍCH / QUYẾT ĐỊNH</span><span>Công cụ hỗ trợ ra quyết định</span></footer>
      </div>
    </main>
  );
}

function translateSegment(segment: string) {
  return { "High intent": "Ý định cao", Consideration: "Đang cân nhắc", "Low intent": "Ý định thấp" }[segment] ?? segment;
}

function translateRisk(risk: string) {
  return { High: "Cao", Medium: "Trung bình", Low: "Thấp" }[risk] ?? risk;
}

function Field({ label, value, suffix, onChange }: { label: string; value: number; suffix: string; onChange: (value: string) => void }) {
  return <label className="block"><span className="mb-2 block text-sm text-slate-300">{label}</span><span className="input-wrap"><input type="number" min="0" value={value} onChange={(event) => onChange(String(Number(event.target.value) || 0))} onBlur={(event) => onChange(String(Number(event.target.value) || 0))} required /><span>{suffix}</span></span></label>;
}

function Metric({ icon, label, value, tone = "" }: { icon: React.ReactNode; label: string; value: string; tone?: string }) {
  return <div className="metric"><span className="text-mint">{icon}</span><p className="mt-3 text-[11px] uppercase tracking-wider text-slate-500">{label}</p><p className={`mt-1 truncate text-sm font-semibold ${tone}`}>{value}</p></div>;
}

export default App;
