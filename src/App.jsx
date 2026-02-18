import { Routes, Route, Navigate } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import History from "./pages/History";
import AnalysisDetail from "./pages/AnalysisDetail";
import About from "./pages/About";
import "./App.css";
export default function App() {
  return (
    <div className="app">
      <Header />
      <section className="disclaimer" role="note" aria-label="Rechtlicher Hinweis">
        <div className="container disclaimer-inner">
          <strong>Hinweis:</strong>&nbsp;Die angezeigten Ergebnisse sind nur eine
          unverbindliche KI-Einschätzung und keine Rechtsberatung.
        </div>
      </section>
      <main className="content">
        <div className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/history" element={<History />} />
            <Route path="/history/:id" element={<AnalysisDetail />} />
            <Route path="*" element={<Navigate to="/" replace />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </main>
      <Footer />
    </div>
  );
}
