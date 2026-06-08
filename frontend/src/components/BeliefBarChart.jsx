import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function BeliefBarChart({ nodes }) {
  if (!nodes || nodes.length === 0) return null

  // Filter to non-leaf nodes only
  const nonLeafNodes = nodes.filter(n => n.children?.length > 0 || n.is_internal || (!n.is_leaf && n.p_true !== undefined))

  const chartData = nonLeafNodes.length > 0 ? nonLeafNodes : nodes

  const data = chartData.map(node => ({
    name: node.name || node.requirement || 'Unknown',
    True: node.p_true ?? node.pTrue ?? 0,
    False: node.p_false ?? node.pFalse ?? 0,
  }))

  return (
    <div style={{ marginTop: '8px' }}>
      <h3 style={{ margin: '0 0 8px 0', fontSize: '16px' }}>Belief Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Bar dataKey="True" fill="#16a34a" />
          <Bar dataKey="False" fill="#dc2626" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
