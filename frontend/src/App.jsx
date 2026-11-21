import { useState } from 'react'
import {
  Activity, BarChart3, Database, LayoutDashboard, Upload,
  Layers3, ScanSearch, Settings, Bell, Search, ChevronDown,
  ArrowUpRight, ArrowDownRight, Cpu, FileImage, CheckCircle2,
  Clock3, Zap, Menu, X, MoreHorizontal, Download
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts'

const spectralData = [
  { band: '400', value: 22 },
  { band: '450', value: 38 },
  { band: '500', value: 32 },
  { band: '550', value: 57 },
  { band: '600', value: 45 },
  { band: '650', value: 72 },
  { band: '700', value: 61 },
  { band: '750', value: 85 },
  { band: '800', value: 68 },
  { band: '850', value: 92 },
  { band: '900', value: 77 },
]

const navigation = [
  { title: 'OVERVIEW', items: [
    { label: 'Dashboard', icon: LayoutDashboard },
    { label: 'Data Explorer', icon: Database },
  ]},
  { title: 'ANALYSIS', items: [
    { label: 'Preprocessing', icon: Layers3 },
    { label: 'Spectral Analysis', icon: ScanSearch },
    { label: 'Feature Dataset', icon: BarChart3 },
  ]},
]

const datasets = [
  { name: 'sample_hyperspectral.tif', type: 'GeoTIFF', size: '676 KB', status: 'Processed', date: 'Nov 20, 2026', color: 'cyan' },
  { name: 'terrain_reflectance.h5', type: 'HDF5', size: '2.4 MB', status: 'Ready', date: 'Nov 19, 2026', color: 'violet' },
  { name: 'vegetation_sample.tif', type: 'GeoTIFF', size: '1.8 MB', status: 'Processing', date: 'Nov 18, 2026', color: 'amber' },
]

function Sidebar({ active, setActive, open, close }) {
  return (
    <>
      {open && <div className="sidebar-overlay" onClick={close} />}
      <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-icon"><Activity size={23} /></div>
          <div>
            <h2>Terra<span>Spectra</span></h2>
            <p>INTELLIGENCE PLATFORM</p>
          </div>
          <button className="mobile-close" onClick={close}><X size={20}/></button>
        </div>

        <div className="workspace">
          <div className="workspace-avatar">TS</div>
          <div><strong>Research Workspace</strong><small>Personal environment</small></div>
          <ChevronDown size={15}/>
        </div>

        {navigation.map(group => (
          <div className="nav-group" key={group.title}>
            <p className="nav-heading">{group.title}</p>
            {group.items.map(item => {
              const Icon = item.icon
              return (
                <button
                  key={item.label}
                  className={`nav-item ${active === item.label ? 'active' : ''}`}
                  onClick={() => { setActive(item.label); close() }}
                >
                  <Icon size={18}/>
                  <span>{item.label}</span>
                  {active === item.label && <span className="active-dot"/>}
                </button>
              )
            })}
          </div>
        ))}

        <div className="sidebar-bottom">
          <div className="storage-card">
            <div className="storage-title"><Database size={16}/> Storage usage</div>
            <div className="storage-bar"><span/></div>
            <p>2.4 GB <span>of 10 GB</span></p>
          </div>
          <button className="nav-item"><Settings size={18}/> Settings</button>
          <div className="profile">
            <div className="profile-avatar">HG</div>
            <div><strong>Researcher</strong><small>Workspace Admin</small></div>
            <MoreHorizontal size={19}/>
          </div>
        </div>
      </aside>
    </>
  )
}

function MetricCard({ icon: Icon, title, value, change, description, color, down }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <div className={`metric-icon ${color}`}><Icon size={20}/></div>
        <button className="more-button"><MoreHorizontal size={19}/></button>
      </div>
      <p className="metric-label">{title}</p>
      <div className="metric-value">{value}</div>
      <div className="metric-footer">
        <span className={`metric-change ${down ? 'negative' : ''}`}>
          {down ? <ArrowDownRight size={14}/> : <ArrowUpRight size={14}/>}
          {change}
        </span>
        <span>{description}</span>
      </div>
    </div>
  )
}

function Dashboard({ active }) {
  if (active !== 'Dashboard') {
    return (
      <div className="placeholder-page">
        <div className="placeholder-icon"><Layers3 size={28}/></div>
        <h2>{active}</h2>
        <p>This workspace is ready for the next development milestone.</p>
        <span>Module integration coming soon</span>
      </div>
    )
  }

  return (
    <>
      <div className="welcome-row">
        <div>
          <div className="eyebrow"><span/> WORKSPACE OVERVIEW</div>
          <h1>Good evening, <span>Researcher</span></h1>
          <p className="page-description">Monitor your hyperspectral data and analysis pipeline.</p>
        </div>
        <button className="date-button"><Clock3 size={15}/> November 21, 2026</button>
      </div>

      <div className="metrics-grid">
        <MetricCard icon={FileImage} title="Total Datasets" value="12" change="+2.5%" description="vs last month" color="cyan"/>
        <MetricCard icon={CheckCircle2} title="Processed Images" value="08" change="+12.8%" description="vs last month" color="green"/>
        <MetricCard icon={BarChart3} title="Spectral Features" value="48K" change="+8.2%" description="vs last month" color="violet"/>
        <MetricCard icon={Cpu} title="Pipeline Runs" value="24" change="-2.4%" description="vs last month" color="orange" down/>
      </div>

      <div className="main-grid">
        <section className="panel spectral-panel">
          <div className="panel-header">
            <div><h3>Spectral Signature</h3><p>Reflectance intensity across wavelength bands</p></div>
            <button className="select-button">Sample dataset <ChevronDown size={15}/></button>
          </div>
          <div className="chart-legend"><span/> Sample reflectance <small>• Demo visualization</small></div>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={spectralData}>
                <defs>
                  <linearGradient id="spectralFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.28}/>
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#252633" strokeDasharray="3 5" vertical={false}/>
                <XAxis dataKey="band" stroke="#77798c" tickLine={false} axisLine={false} tick={{fontSize:11}}/>
                <YAxis stroke="#77798c" tickLine={false} axisLine={false} tick={{fontSize:11}}/>
                <Tooltip contentStyle={{background:'#171823',border:'1px solid #303144',borderRadius:10,color:'#fff'}}/>
                <Area type="monotone" dataKey="value" stroke="#22d3ee" strokeWidth={2.5} fill="url(#spectralFill)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-axis-labels"><span>Wavelength (nm)</span><span>Reflectance (%)</span></div>
        </section>

        <section className="panel pipeline-panel">
          <div className="panel-header"><div><h3>Processing Pipeline</h3><p>Analysis workflow overview</p></div><button className="more-button"><MoreHorizontal size={20}/></button></div>
          <div className="pipeline-list">
            {[
              ['Data Ingestion','Import hyperspectral imagery','done'],
              ['Normalization','Missing value handling','done'],
              ['PCA Reduction','Dimensionality reduction','done'],
              ['Feature Extraction','Spectral feature generation','done'],
              ['ML Classification','Ground-truth labels required','pending'],
            ].map(([title,sub,status],i)=>(
              <div className="pipeline-step" key={title}>
                <div className={`step-icon ${status}`}><span>{status === 'done' ? <CheckCircle2 size={17}/> : <Clock3 size={17}/>}</span></div>
                <div className="step-copy"><strong>{title}</strong><small>{sub}</small></div>
                <span className={`step-number ${status}`}>{status === 'done' ? 'Done' : 'Pending'}</span>
              </div>
            ))}
          </div>
          <div className="pipeline-note"><Zap size={16}/><span>4 of 5 pipeline stages completed</span></div>
        </section>
      </div>

      <section className="panel datasets-panel">
        <div className="panel-header">
          <div><h3>Recent Datasets</h3><p>Your latest hyperspectral data activity</p></div>
          <button className="view-all" onClick={() => window.alert('Dataset explorer will be connected soon.')}>View all <ArrowUpRight size={15}/></button>
        </div>
        <div className="table-scroll">
          <table>
            <thead><tr><th>DATASET NAME</th><th>FORMAT</th><th>SIZE</th><th>STATUS</th><th>LAST UPDATED</th><th/></tr></thead>
            <tbody>
              {datasets.map(item=>(
                <tr key={item.name}>
                  <td><div className="dataset-name"><div className={`file-icon ${item.color}`}><FileImage size={17}/></div><strong>{item.name}</strong></div></td>
                  <td>{item.type}</td><td>{item.size}</td>
                  <td><span className={`status ${item.color}`}>{item.status}</span></td>
                  <td>{item.date}</td>
                  <td><button className="more-button"><MoreHorizontal size={18}/></button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="bottom-note"><Activity size={15}/> TerraSpectra Intelligence Platform <span>•</span> Frontend Preview v1.0</div>
    </>
  )
}

function App() {
  const [active, setActive] = useState('Dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="app-shell">
      <Sidebar active={active} setActive={setActive} open={sidebarOpen} close={()=>setSidebarOpen(false)}/>
      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu" onClick={()=>setSidebarOpen(true)}><Menu size={21}/></button>
          <div className="breadcrumb">Workspace <span>/</span> <strong>{active}</strong></div>
          <div className="topbar-actions">
            <div className="search-box"><Search size={16}/><input placeholder="Search anything..."/><kbd>⌘ K</kbd></div>
            <button className="icon-button" aria-label="Notifications"><Bell size={19}/><i/></button>
            <div className="top-avatar">HG</div>
          </div>
        </header>
        <div className="page-content">
          <Dashboard active={active}/>
        </div>
      </main>
    </div>
  )
}

export default App
