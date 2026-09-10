import { StatusBar } from "expo-status-bar";
import { useState } from "react";
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

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

const API_URL = "http://127.0.0.1:8000/predict";
const initialForm: FormState = { views: 14, cart_additions: 3, total_spent: 185, days_since_last_active: 6 };
const funnelData = [
  ["Lượt xem sản phẩm", 86],
  ["Thêm vào giỏ", 54],
  ["Bắt đầu thanh toán", 38],
  ["Mua hàng", 24],
] as const;

export default function App() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (field: keyof FormState, value: string) => {
    setForm((current) => ({ ...current, [field]: Number(value) || 0 }));
  };

  const predict = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) throw new Error("Prediction failed");
      setPrediction((await response.json()) as Prediction);
    } catch {
      setError("Không thể kết nối API. Hãy khởi động FastAPI trên cổng 8000.");
    } finally {
      setLoading(false);
    }
  };

  const probability = prediction ? Math.round(prediction.purchase_probability * 100) : 0;
  const segment = prediction ? translateSegment(prediction.segment) : "Chưa có dữ liệu";
  const risk = prediction ? translateRisk(prediction.churn_risk) : "Chưa có dữ liệu";

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <View style={styles.brandRow}>
          <View style={styles.logo}><Text style={styles.logoText}>⌁</Text></View>
          <View><Text style={styles.brand}>PULSEBOARD</Text><Text style={styles.eyebrow}>CUSTOMER BEHAVIOR / 02</Text></View>
        </View>
        <Text style={styles.online}>● MÔ HÌNH ĐANG HOẠT ĐỘNG</Text>
      </View>

      <View style={styles.intro}>
        <Text style={styles.eyebrowMint}>CUSTOMER BEHAVIOR DASHBOARD</Text>
        <Text style={styles.title}>Khả năng mua hàng</Text>
        <Text style={styles.description}>Nhập hoạt động của khách hàng để ước tính khả năng mua và nguy cơ rời bỏ.</Text>
      </View>

      <View style={styles.panel}>
        <SectionTitle index="01 / DỮ LIỆU ĐẦU VÀO" title="Thông tin khách hàng" />
        <Field label="Lượt xem sản phẩm" suffix="lượt" value={form.views} onChange={(value) => updateField("views", value)} />
        <Field label="Số lần thêm vào giỏ" suffix="lần" value={form.cart_additions} onChange={(value) => updateField("cart_additions", value)} />
        <Field label="Tổng chi tiêu" suffix="USD" value={form.total_spent} onChange={(value) => updateField("total_spent", value)} />
        <Field label="Số ngày không hoạt động" suffix="ngày" value={form.days_since_last_active} onChange={(value) => updateField("days_since_last_active", value)} />
        <View style={styles.actions}>
          <Pressable style={styles.primaryButton} onPress={predict} disabled={loading}><Text style={styles.primaryText}>{loading ? "Đang phân tích..." : "◎  Dự đoán  ↗"}</Text></Pressable>
          <Pressable style={styles.resetButton} onPress={() => { setForm(initialForm); setPrediction(null); setError(""); }}><Text style={styles.resetText}>↻</Text></Pressable>
        </View>
        {error ? <Text style={styles.error}>{error}</Text> : null}
      </View>

      <View style={styles.panel}>
        <SectionTitle index="02 / KẾT QUẢ MÔ HÌNH" title="Khả năng mua hàng" />
        <View style={styles.probabilityRow}><Text style={styles.probability}>{probability}<Text style={styles.percent}>%</Text></Text><View style={styles.progressBox}><Text style={styles.muted}>Xác suất mua hàng</Text><View style={styles.progressTrack}><View style={[styles.progressValue, { width: `${probability}%` }]} /></View></View></View>
        <View style={styles.metrics}><Metric label="PHÂN NHÓM" value={segment} /><Metric label="NGUY CƠ RỜI BỎ" value={risk} /><Metric label="MÔ HÌNH" value="Random Forest" /></View>
      </View>

      <View style={styles.panel}>
        <View style={styles.chartHeader}><SectionTitle index="03 / TỔNG QUAN" title="Phân tích phễu chuyển đổi" /><Text style={styles.period}>30 ngày qua</Text></View>
        {funnelData.map(([label, value]) => <View style={styles.barRow} key={label}><Text style={styles.barLabel}>{label}</Text><View style={styles.barTrack}><View style={[styles.bar, { width: `${value}%` }]} /></View></View>)}
        <Text style={styles.chartNote}>● Tỷ lệ chuyển đổi theo hành động</Text>
      </View>
      <Text style={styles.footer}>DỮ LIỆU / PHÂN TÍCH / QUYẾT ĐỊNH</Text>
    </ScrollView>
  );
}

function SectionTitle({ index, title }: { index: string; title: string }) { return <View style={styles.sectionTitle}><Text style={styles.eyebrow}>{index}</Text><Text style={styles.heading}>{title}</Text></View>; }
function Field({ label, suffix, value, onChange }: { label: string; suffix: string; value: number; onChange: (value: string) => void }) { return <View style={styles.field}><Text style={styles.label}>{label}</Text><View style={styles.inputRow}><TextInput style={styles.input} keyboardType="number-pad" value={String(value)} onChangeText={onChange} onBlur={() => onChange(String(value))} /><Text style={styles.suffix}>{suffix}</Text></View></View>; }
function Metric({ label, value }: { label: string; value: string }) { return <View style={styles.metric}><Text style={styles.metricIcon}>◌</Text><Text style={styles.metricLabel}>{label}</Text><Text style={styles.metricValue} numberOfLines={1}>{value}</Text></View>; }
function translateSegment(value: string) { return ({ "High intent": "Ý định cao", Consideration: "Đang cân nhắc", "Low intent": "Ý định thấp" } as Record<string, string>)[value] ?? value; }
function translateRisk(value: string) { return ({ High: "Cao", Medium: "Trung bình", Low: "Thấp" } as Record<string, string>)[value] ?? value; }

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: "#f4f7f6" },
  content: { padding: 18, paddingTop: 24, paddingBottom: 32 },
  header: { borderBottomWidth: 1, borderBottomColor: "#dce5e2", paddingBottom: 18, flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  brandRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  logo: { width: 34, height: 34, borderRadius: 7, backgroundColor: "#8ce3c3", alignItems: "center", justifyContent: "center" },
  logoText: { fontSize: 23, color: "#17201e", fontWeight: "700" },
  brand: { color: "#1f2c29", fontSize: 14, fontWeight: "700", letterSpacing: 0.5 },
  online: { color: "#329d79", fontSize: 8, fontWeight: "700", maxWidth: 120, textAlign: "right" },
  eyebrow: { color: "#71807c", fontSize: 8, fontWeight: "700", letterSpacing: 1.1 },
  eyebrowMint: { color: "#328b70", fontSize: 9, fontWeight: "700", letterSpacing: 1.1, marginBottom: 8 },
  intro: { paddingVertical: 26 },
  title: { color: "#1f2c29", fontSize: 30, fontWeight: "700", letterSpacing: -0.6 },
  description: { color: "#64716f", fontSize: 12, lineHeight: 19, marginTop: 8 },
  panel: { backgroundColor: "#fff", borderWidth: 1, borderColor: "#dce5e2", borderRadius: 8, padding: 18, marginBottom: 16, shadowColor: "#243b35", shadowOpacity: 0.06, shadowRadius: 8, shadowOffset: { width: 0, height: 2 }, elevation: 1 },
  sectionTitle: { flex: 1, marginBottom: 18 },
  heading: { color: "#1f2c29", fontSize: 19, fontWeight: "700", marginTop: 6 },
  field: { marginBottom: 14 },
  label: { color: "#42524e", fontSize: 12, marginBottom: 6 },
  inputRow: { height: 43, borderWidth: 1, borderColor: "#cbd8d4", borderRadius: 6, backgroundColor: "#fbfdfc", flexDirection: "row", alignItems: "center" },
  input: { flex: 1, paddingHorizontal: 12, color: "#1f2c29", fontSize: 13, fontWeight: "600" },
  suffix: { color: "#71807c", fontSize: 10, paddingRight: 12 },
  actions: { flexDirection: "row", gap: 9, marginTop: 4 },
  primaryButton: { backgroundColor: "#70c9aa", borderRadius: 6, paddingVertical: 12, paddingHorizontal: 15 },
  primaryText: { color: "#17201e", fontSize: 12, fontWeight: "700" },
  resetButton: { width: 43, borderWidth: 1, borderColor: "#cbd8d4", borderRadius: 6, alignItems: "center", justifyContent: "center" },
  resetText: { color: "#64716f", fontSize: 20 },
  error: { color: "#a33d35", backgroundColor: "#fff0ee", padding: 10, borderRadius: 5, fontSize: 11, marginTop: 14 },
  probabilityRow: { flexDirection: "row", alignItems: "flex-end", gap: 14, marginTop: 3, marginBottom: 23 },
  probability: { color: "#70c9aa", fontSize: 76, lineHeight: 79, fontWeight: "600", letterSpacing: -4 },
  percent: { fontSize: 23, letterSpacing: 0 },
  progressBox: { flex: 1, paddingBottom: 6 },
  muted: { color: "#71807c", fontSize: 11, marginBottom: 7 },
  progressTrack: { height: 5, backgroundColor: "#dce5e2", borderRadius: 4, overflow: "hidden" },
  progressValue: { height: 5, backgroundColor: "#51b892", borderRadius: 4 },
  metrics: { flexDirection: "row", gap: 7 },
  metric: { flex: 1, minWidth: 0, borderWidth: 1, borderColor: "#dce5e2", borderRadius: 5, backgroundColor: "#f7faf9", padding: 9 },
  metricIcon: { color: "#328b70", fontSize: 14 },
  metricLabel: { color: "#71807c", fontSize: 8, fontWeight: "700", marginTop: 8 },
  metricValue: { color: "#1f2c29", fontSize: 10, fontWeight: "600", marginTop: 4 },
  chartHeader: { flexDirection: "row" },
  period: { color: "#71807c", borderWidth: 1, borderColor: "#dce5e2", borderRadius: 12, paddingHorizontal: 9, paddingVertical: 6, fontSize: 9, height: 27 },
  barRow: { flexDirection: "row", alignItems: "center", marginBottom: 15 },
  barLabel: { color: "#64716f", fontSize: 10, width: 112 },
  barTrack: { flex: 1, height: 20, backgroundColor: "#eef3f1", borderRadius: 3, overflow: "hidden" },
  bar: { height: 20, backgroundColor: "#51b892", borderRadius: 3 },
  chartNote: { color: "#71807c", fontSize: 10, marginTop: 2 },
  footer: { color: "#899692", fontSize: 8, letterSpacing: 0.5, marginTop: 2 },
});
