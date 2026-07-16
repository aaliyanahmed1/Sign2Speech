import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts';
import { Activity, Zap, MessageSquare, TrendingUp, RefreshCw } from 'lucide-react';
import { apiClient } from '../lib/api';
import type { AnalyticsResponse } from '../types/api';

const COLORS = ['#4f46e5', '#0284c7', '#db2777', '#f59e0b', '#10b981', '#6366f1', '#a855f7', '#f97316'];

const PLACEHOLDER_FREQ = [
  { name: 'school', count: 42 },
  { name: 'sorry', count: 28 },
  { name: 'help', count: 35 },
  { name: 'friend', count: 22 },
  { name: 'teacher', count: 31 },
  { name: 'good', count: 45 },
  { name: 'home', count: 18 },
  { name: 'work', count: 25 },
];

const PLACEHOLDER_DETECTIONS = [
  { gesture: 'good', confidence: 0.92, timestamp: '2024-01-15 10:23:00' },
  { gesture: 'school', confidence: 0.87, timestamp: '2024-01-15 10:22:45' },
  { gesture: 'help', confidence: 0.95, timestamp: '2024-01-15 10:22:30' },
  { gesture: 'teacher', confidence: 0.81, timestamp: '2024-01-15 10:22:15' },
  { gesture: 'friend', confidence: 0.78, timestamp: '2024-01-15 10:22:00' },
];

export default function Analytics() {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const result = await apiClient.getAnalytics();
      setData(result);
    } catch {
      // use placeholder
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const freqData = data?.gesture_frequency
    ? Object.entries(data.gesture_frequency)
        .map(([name, count]) => ({ name, count }))
        .filter((d) => d.count > 0)
        .sort((a, b) => b.count - a.count)
    : PLACEHOLDER_FREQ;

  const detections = data?.recent_detections?.length
    ? data.recent_detections.slice(0, 10)
    : PLACEHOLDER_DETECTIONS;

  const confidenceOverTime = detections.map((d, i) => ({
    idx: i + 1,
    confidence: d.confidence,
    name: d.gesture,
  }));

  const pieData = freqData.slice(0, 8);

  return (
    <div className="max-w-7xl mx-auto p-6 md:p-8 font-sans space-y-8">
      <div className="flex items-center justify-between border-b border-outline-variant pb-6 gap-4">
        <div>
          <h2 className="text-3xl font-syne font-bold text-on-surface">Analytics Dashboard</h2>
          <p className="text-on-surface-variant text-sm mt-1.5">Real-time model performance and gesture statistics</p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="p-2 border border-outline-variant hover:border-outline rounded-xl hover:bg-surface-variant transition-colors text-on-surface-variant hover:text-on-surface cursor-pointer"
          aria-label="Refresh analytics"
        >
          <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { icon: <Activity size={20} />, label: 'Sessions', value: data?.total_sessions ?? 0, color: 'var(--color-primary)' },
          { icon: <Zap size={20} />, label: 'Gestures Detected', value: data?.gestures_detected ?? 0, color: 'var(--color-secondary)' },
          { icon: <MessageSquare size={20} />, label: 'Sentences Spoken', value: data?.sentences_spoken ?? 0, color: 'var(--color-tertiary)' },
          { icon: <TrendingUp size={20} />, label: 'Avg Confidence', value: data ? `${(data.avg_confidence * 100).toFixed(0)}%` : '92%', color: '#f59e0b' },
        ].map((s) => (
          <div key={s.label} className="bg-surface rounded-2xl p-5 border border-outline-variant shadow-sm space-y-3 text-left">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center border border-outline-variant" style={{ background: 'var(--color-surface-variant)', color: s.color }}>
                {s.icon}
              </div>
              <span className="text-xs text-on-surface-variant font-mono uppercase tracking-wider">{s.label}</span>
            </div>
            <div className="text-3xl font-syne font-bold text-on-surface">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Charts grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Gesture frequency bar chart */}
        <div className="bg-surface rounded-2xl p-6 border border-outline-variant shadow-sm">
          <h3 className="text-sm font-syne font-bold text-on-surface-variant uppercase tracking-wider mb-5 text-left">Gesture Frequency</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={freqData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-outline-variant)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--color-on-surface-variant)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--color-on-surface-variant)', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'var(--color-surface)', border: '1px solid var(--color-outline-variant)', borderRadius: 12, fontSize: 12 }}
                labelStyle={{ color: 'var(--color-on-surface-variant)' }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {freqData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie chart */}
        <div className="bg-surface rounded-2xl p-6 border border-outline-variant shadow-sm">
          <h3 className="text-sm font-syne font-bold text-on-surface-variant uppercase tracking-wider mb-5 text-left">Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={3}
                dataKey="count"
                nameKey="name"
                label={({ name, percent }: { name?: string; percent?: number }) =>
                  `${name ?? ''} ${((percent ?? 0) * 100).toFixed(0)}%`
                }
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: 'var(--color-surface)', border: '1px solid var(--color-outline-variant)', borderRadius: 12, fontSize: 12 }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Confidence over time */}
        <div className="bg-surface rounded-2xl p-6 border border-outline-variant shadow-sm">
          <h3 className="text-sm font-syne font-bold text-on-surface-variant uppercase tracking-wider mb-5 text-left">Confidence Trend</h3>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={confidenceOverTime}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-outline-variant)" />
              <XAxis dataKey="idx" tick={{ fill: 'var(--color-on-surface-variant)', fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: 'var(--color-on-surface-variant)', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'var(--color-surface)', border: '1px solid var(--color-outline-variant)', borderRadius: 12, fontSize: 12 }}
              />
              <defs>
                <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="confidence" stroke="var(--color-primary)" fill="url(#grad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Recent detections table */}
        <div className="bg-surface rounded-2xl p-6 border border-outline-variant shadow-sm">
          <h3 className="text-sm font-syne font-bold text-on-surface-variant uppercase tracking-wider mb-5 text-left">Recent Detections</h3>
          <div className="overflow-y-auto max-h-[280px]">
            <table className="w-full text-sm border-collapse text-left">
              <thead>
                <tr className="text-on-surface-variant text-xs border-b border-outline-variant font-mono">
                  <th className="pb-3">Gesture</th>
                  <th className="pb-3 text-center">Confidence</th>
                  <th className="pb-3 text-right">Time</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((d, i) => (
                  <tr key={i} className="border-b border-outline-variant/55 hover:bg-surface-variant/40 transition-colors">
                    <td className="py-3.5 text-primary font-mono font-bold">{d.gesture}</td>
                    <td className="py-3.5 text-center font-mono">
                      <span className={`font-bold ${(d.confidence * 100) >= 80 ? 'text-primary' : (d.confidence * 100) >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {(d.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="py-3.5 text-right text-on-surface-variant text-xs">{d.timestamp}</td>
                  </tr>
                ))}
                {detections.length === 0 && (
                  <tr><td colSpan={3} className="py-8 text-center text-on-surface-variant italic">No detections yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
