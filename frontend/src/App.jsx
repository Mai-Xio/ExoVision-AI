import { useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  Brain,
  CheckCircle,
  Database,
  Gauge,
  Orbit,
  Search,
  Server,
  ShieldCheck,
  Sparkles,
  Star,
  Target,
  Telescope,
  XCircle,
} from "lucide-react";
import {
  BarChart,
  Bar,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [health, setHealth] = useState(null);
  const [model, setModel] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [families, setFamilies] = useState([]);

  const [searchName, setSearchName] = useState("K02717.01");
  const [screening, setScreening] = useState(null);
  const [screeningLoading, setScreeningLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    loadDashboard();
  }, []);

  async function loadDashboard() {
    try {
      const [
        healthResponse,
        modelResponse,
        candidatesResponse,
        familiesResponse,
      ] = await Promise.all([
        axios.get(`${API}/health`),
        axios.get(`${API}/api/model/performance`),
        axios.get(`${API}/api/candidates`),
        axios.get(`${API}/api/explainability/families`),
      ]);

      setHealth(healthResponse.data);
      setModel(modelResponse.data);
      const candidateData = candidatesResponse.data;

setCandidates(
  Array.isArray(candidateData)
    ? candidateData
    : candidateData.candidates || candidateData.results || []
);
      setFamilies(familiesResponse.data);
    } catch (err) {
      console.error(err);
      setError(
        "Could not connect to the ExoVision AI backend. Make sure FastAPI is running on port 8000."
      );
    }
  }

  async function screenCandidate() {
    if (!searchName.trim()) return;

    setScreeningLoading(true);
    setScreening(null);
    setError("");

    try {
      const response = await axios.post(
        `${API}/api/candidates/${searchName.trim()}/screen`
      );

      setScreening(response.data);
    } catch (err) {
      console.error(err);

      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError("Candidate could not be screened.");
      }
    } finally {
      setScreeningLoading(false);
    }
  }

  return (
    <div className="app">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        health={health}
      />

      <main className="main">
        <Header activePage={activePage} />

        {error && (
          <div className="error-banner">
            <XCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {activePage === "Dashboard" && (
          <Dashboard
            health={health}
            model={model}
            candidates={candidates}
            setActivePage={setActivePage}
          />
        )}

        {activePage === "Candidates" && (
          <Candidates candidates={candidates} />
        )}

        {activePage === "AI Screening" && (
          <AIScreening
            searchName={searchName}
            setSearchName={setSearchName}
            screening={screening}
            loading={screeningLoading}
            screenCandidate={screenCandidate}
          />
        )}

        {activePage === "Model" && <Model model={model} />}

        {activePage === "Explainability" && (
          <Explainability families={families} />
        )}
      </main>
    </div>
  );
}

function Sidebar({ activePage, setActivePage, health }) {
  const navigation = [
    { name: "Dashboard", icon: Gauge },
    { name: "Candidates", icon: Star },
    { name: "AI Screening", icon: Brain },
    { name: "Model", icon: Activity },
    { name: "Explainability", icon: Target },
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <Telescope size={25} />
        </div>

        <div>
          <div className="brand-name">ExoVision AI</div>
          <div className="brand-subtitle">EXOPLANET ANALYTICS</div>
        </div>
      </div>

      <div className="nav-section">
        <div className="nav-label">WORKSPACE</div>

        {navigation.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.name}
              className={`nav-item ${
                activePage === item.name ? "active" : ""
              }`}
              onClick={() => setActivePage(item.name)}
            >
              <Icon size={18} />
              <span>{item.name}</span>
            </button>
          );
        })}
      </div>

      <div className="sidebar-bottom">
        <div className="system-card">
          <div className="system-title">
            <Server size={16} />
            SYSTEM STATUS
          </div>

          <div className="system-status">
            <span className="status-dot"></span>
            {health?.status === "healthy" ? "API ONLINE" : "CONNECTING"}
          </div>

          <div className="system-detail">
            Random Forest • 35 features
          </div>
        </div>
      </div>
    </aside>
  );
}

function Header({ activePage }) {
  return (
    <header className="header">
      <div>
        <div className="breadcrumb">EXOVISION AI / {activePage.toUpperCase()}</div>
        <h1>{activePage}</h1>
      </div>

      <div className="header-status">
        <span className="status-dot"></span>
        LIVE MODEL
      </div>
    </header>
  );
}

function Dashboard({ health, model, candidates, setActivePage }) {
  const rf = model?.random_forest?.test;

  return (
    <div className="page">
      <section className="hero">
        <div>
          <div className="eyebrow">
            <Sparkles size={15} />
            MACHINE LEARNING • KEPLER DATA
          </div>

          <h2>
            Detecting
            <span> Earth-like worlds</span>
            <br />
            in stellar transit data.
          </h2>

          <p>
            Physics-informed machine learning for screening Kepler Objects of
            Interest and identifying candidates that satisfy project-defined
            Earth-like criteria.
          </p>

          <button
            className="primary-button"
            onClick={() => setActivePage("AI Screening")}
          >
            <Search size={17} />
            Screen a Candidate
          </button>
        </div>

        <div className="hero-orbit">
          <div className="orbit orbit-one"></div>
          <div className="orbit orbit-two"></div>
          <div className="planet"></div>
          <div className="star"></div>
        </div>
      </section>

      <section className="stats-grid">
        <StatCard
          icon={<Database size={20} />}
          label="KEPLER OBJECTS"
          value={health?.candidate_count ? "3,811" : "—"}
          detail="Training dataset"
        />

        <StatCard
          icon={<Target size={20} />}
          label="SCREENED CANDIDATES"
          value={candidates.length || "—"}
          detail="Current project screen"
        />

        <StatCard
          icon={<Activity size={20} />}
          label="TEST ROC-AUC"
          value={rf?.roc_auc ? rf.roc_auc.toFixed(4) : "—"}
          detail="Random Forest"
        />

        <StatCard
          icon={<ShieldCheck size={20} />}
          label="TEST ACCURACY"
          value={rf?.accuracy ? `${(rf.accuracy * 100).toFixed(1)}%` : "—"}
          detail="Held-out test set"
        />
      </section>

      <section className="content-grid">
        <div className="panel large-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">CANDIDATE CATALOG</div>
              <h3>Earth-like screening results</h3>
            </div>

            <button
              className="text-button"
              onClick={() => setActivePage("Candidates")}
            >
              View all
            </button>
          </div>

          <CandidateTable candidates={candidates.slice(0, 6)} />
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">MODEL HEALTH</div>
              <h3>Performance</h3>
            </div>
          </div>

          <PerformanceMetrics model={model} />
        </div>
      </section>
    </div>
  );
}

function StatCard({ icon, label, value, detail }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>

      <div className="stat-label">{label}</div>

      <div className="stat-value">{value}</div>

      <div className="stat-detail">{detail}</div>
    </div>
  );
}

function CandidateTable({ candidates }) {
  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>KOI</th>
            <th>PROBABILITY</th>
            <th>RADIUS</th>
            <th>INSOLATION</th>
            <th>SCORE</th>
            <th>PHYSICS</th>
          </tr>
        </thead>

        <tbody>
          {candidates.map((candidate) => (
            <tr key={candidate.kepoi_name}>
              <td className="candidate-name">
                {candidate.kepoi_name}
              </td>

              <td>
                <Probability value={candidate.planet_probability} />
              </td>

              <td>
                {Number(candidate.koi_prad).toFixed(3)} R⊕
              </td>

              <td>
                {Number(candidate.koi_insol).toFixed(3)} S⊕
              </td>

              <td>
                <strong>
                  {Number(candidate.earthlike_score).toFixed(3)}
                </strong>
              </td>

              <td>
                {candidate.overall_physics_consistent ? (
                  <span className="pass">
                    <CheckCircle size={15} />
                    PASS
                  </span>
                ) : (
                  <span className="fail">
                    <XCircle size={15} />
                    REVIEW
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Probability({ value }) {
  return (
    <div className="probability">
      <div className="probability-bar">
        <div style={{ width: `${value * 100}%` }}></div>
      </div>

      <span>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function PerformanceMetrics({ model }) {
  const rf = model?.random_forest?.test;

  if (!rf) {
    return <div className="loading">Loading model metrics...</div>;
  }

  const metrics = [
    ["Accuracy", rf.accuracy],
    ["Precision", rf.precision],
    ["Recall", rf.recall],
    ["F1", rf.f1],
    ["ROC-AUC", rf.roc_auc],
    ["PR-AUC", rf.pr_auc],
    ["MCC", rf.mcc],
  ];

  return (
    <div className="metrics">
      {metrics.map(([label, value]) => (
        <div className="metric-row" key={label}>
          <span>{label}</span>

          <div className="metric-track">
            <div style={{ width: `${value * 100}%` }}></div>
          </div>

          <strong>{value.toFixed(4)}</strong>
        </div>
      ))}
    </div>
  );
}

function Candidates({ candidates }) {
  return (
    <div className="page">
      <div className="page-intro">
        <div className="eyebrow">
          <Star size={15} />
          KEPLER CANDIDATE CATALOG
        </div>

        <h2>Earth-like screened candidates</h2>

        <p>
          Candidates passing the project's radius, insolation, and machine
          learning probability thresholds.
        </p>
      </div>

      <div className="panel">
        <CandidateTable candidates={candidates} />
      </div>
    </div>
  );
}

function AIScreening({
  searchName,
  setSearchName,
  screening,
  loading,
  screenCandidate,
}) {
  return (
    <div className="page">
      <div className="page-intro">
        <div className="eyebrow">
          <Brain size={15} />
          LIVE MODEL INFERENCE
        </div>

        <h2>AI Candidate Screening</h2>

        <p>
          Enter a Kepler Object of Interest to run it through the trained
          Random Forest model and the project's physics-informed screening
          layer.
        </p>
      </div>

      <div className="search-panel">
        <div className="search-input-wrapper">
          <Search size={20} />

          <input
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
            placeholder="Example: K02717.01"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                screenCandidate();
              }
            }}
          />
        </div>

        <button
          className="primary-button"
          onClick={screenCandidate}
          disabled={loading}
        >
          {loading ? "Screening..." : "Run AI Screening"}
        </button>
      </div>

      {screening && <ScreeningResult result={screening} />}
    </div>
  );
}

function ScreeningResult({ result }) {
  const criteria = result.screening_criteria;
  const physics = result.physics_consistency;
  const scores = result.scores;
  const physical = result.physical_values;

  return (
    <div className="screening-result">
      <div className="result-header">
        <div>
          <div className="panel-kicker">SCREENING RESULT</div>
          <h3>{result.kepoi_name}</h3>
        </div>

        <div
          className={`result-badge ${
            result.passes_screen ? "result-pass" : "result-review"
          }`}
        >
          {result.passes_screen ? (
            <>
              <CheckCircle size={17} />
              PASSES SCREEN
            </>
          ) : (
            <>
              <XCircle size={17} />
              DOES NOT PASS
            </>
          )}
        </div>
      </div>

      <div className="result-score">
        <div>
          <div className="score-label">EARTH-LIKE SCREENING SCORE</div>
          <div className="big-score">
            {scores.earthlike_screening_score.toFixed(3)}
          </div>
        </div>

        <div className="probability-circle">
          <div>{(result.planet_probability * 100).toFixed(1)}%</div>
          <span>ML PROBABILITY</span>
        </div>
      </div>

      <div className="result-grid">
        <ValueCard
          label="Radius"
          value={`${physical.radius_earth_radii.toFixed(3)} R⊕`}
          pass={criteria.radius.passes}
        />

        <ValueCard
          label="Insolation"
          value={`${physical.insolation_earth_flux.toFixed(3)} S⊕`}
          pass={criteria.insolation.passes}
        />

        <ValueCard
          label="Equilibrium Temperature"
          value={`${physical.equilibrium_temperature_k.toFixed(1)} K`}
        />

        <ValueCard
          label="Orbital Period"
          value={`${physical.orbital_period_days.toFixed(2)} days`}
        />

        <ValueCard
          label="Stellar Temperature"
          value={`${physical.stellar_temperature_k.toFixed(1)} K`}
        />

        <ValueCard
          label="Density Consistency"
          value={physics.density_consistent ? "PASS" : "REVIEW"}
          pass={physics.density_consistent}
        />

        <ValueCard
          label="Duration Consistency"
          value={physics.duration_consistent ? "PASS" : "REVIEW"}
          pass={physics.duration_consistent}
        />

        <ValueCard
          label="Depth Consistency"
          value={physics.depth_consistent ? "PASS" : "REVIEW"}
          pass={physics.depth_consistent}
        />
      </div>

      <div className="consistency-panel">
        <div>
          <div className="panel-kicker">TRANSIT CONSISTENCY</div>

          <div className="consistency-score">
            {scores.transit_consistency_score.toFixed(3)}
          </div>
        </div>

        <div className="consistency-details">
          <span>
            Density ratio: {physical.density_ratio.toFixed(3)}
          </span>

          <span>
            Duration ratio: {physical.duration_ratio.toFixed(3)}
          </span>

          <span>
            Depth ratio: {physical.depth_ratio.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

function ValueCard({ label, value, pass }) {
  return (
    <div className="value-card">
      <span>{label}</span>

      <strong>{value}</strong>

      {pass !== undefined && (
        <small className={pass ? "pass-text" : "fail-text"}>
          {pass ? "Within criterion" : "Requires review"}
        </small>
      )}
    </div>
  );
}

function Model({ model }) {
  if (!model) {
    return (
      <div className="page">
        <div className="loading">Loading model information...</div>
      </div>
    );
  }

  const rf = model.random_forest?.test;
  const gb = model.gradient_boosting?.test;

  return (
    <div className="page">
      <div className="page-intro">
        <div className="eyebrow">
          <Activity size={15} />
          MODEL EVALUATION
        </div>

        <h2>Model Performance</h2>

        <p>
          Held-out test performance from the final physics-feature model
          evaluation.
        </p>
      </div>

      <div className="model-grid">
        <ModelCard
          name="Random Forest"
          metrics={rf}
          primary
        />

        <ModelCard
          name="Gradient Boosting"
          metrics={gb}
        />
      </div>

      <div className="panel methodology">
        <div className="panel-kicker">METHODOLOGY</div>

        <h3>Physics-informed grouped evaluation</h3>

        <p>
          The final model uses 35 retained physics and engineered features
          after removing three zero-variance features. Evaluation uses
          host-star grouping to prevent objects from the same host from
          appearing across validation boundaries.
        </p>
      </div>
    </div>
  );
}

function ModelCard({ name, metrics, primary }) {
  if (!metrics) return null;

  return (
    <div className={`model-card ${primary ? "primary-model" : ""}`}>
      <div className="model-card-header">
        <div className="model-icon">
          <Brain size={21} />
        </div>

        <div>
          <div className="panel-kicker">CLASSIFIER</div>
          <h3>{name}</h3>
        </div>
      </div>

      <div className="model-metrics">
        <ModelMetric label="Accuracy" value={metrics.accuracy} />
        <ModelMetric label="Precision" value={metrics.precision} />
        <ModelMetric label="Recall" value={metrics.recall} />
        <ModelMetric label="F1" value={metrics.f1} />
        <ModelMetric label="ROC-AUC" value={metrics.roc_auc} />
        <ModelMetric label="PR-AUC" value={metrics.pr_auc} />
        <ModelMetric label="MCC" value={metrics.mcc} />
        <ModelMetric label="Brier" value={metrics.brier} decimals={4} />
      </div>
    </div>
  );
}

function ModelMetric({ label, value, decimals = 4 }) {
  return (
    <div className="model-metric">
      <span>{label}</span>
      <strong>{value.toFixed(decimals)}</strong>
    </div>
  );
}

function Explainability({ families }) {
  const chartData = families.map((item) => ({
    family: item.family,
    importance: item.importance,
  }));

  return (
    <div className="page">
      <div className="page-intro">
        <div className="eyebrow">
          <Target size={15} />
          MODEL EXPLAINABILITY
        </div>

        <h2>Feature importance</h2>

        <p>
          Relative Random Forest feature-family importance from the final
          trained model.
        </p>
      </div>

      <div className="panel chart-panel">
        <div className="panel-header">
          <div>
            <div className="panel-kicker">FEATURE FAMILIES</div>
            <h3>Importance distribution</h3>
          </div>
        </div>

        <div className="chart">
          <ResponsiveContainer width="100%" height={420}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{
                top: 10,
                right: 30,
                left: 50,
                bottom: 10,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />

              <XAxis type="number" />

              <YAxis
                type="category"
                dataKey="family"
                width={130}
              />

              <Tooltip />

              <Bar
                dataKey="importance"
                fill="currentColor"
                radius={[0, 5, 5, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel">
        <div className="panel-kicker">INTERPRETATION</div>

        <h3>How to read this</h3>

        <p className="explanation">
          Feature importance describes how much the trained Random Forest used
          features for prediction. It should not be interpreted as causal
          evidence. Several engineered features are correlated or derived
          from the same physical quantities.
        </p>
      </div>
    </div>
  );
}

export default App;