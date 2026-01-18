export default function About() {
  return (
    <div className="container">
      <div className="card">
        <div className="card-inner">
          <h1 className="h1">Über das Projekt</h1>

          <p className="p">
            <strong>VerdictIQ</strong> ist eine Web-Applikation zur
            automatisierten Analyse von Gerichtsurteilen. Ziel ist es,
            juristische Texte strukturiert auszuwerten und eine erste
            Einschätzung zu relevanten Fragestellungen (z.B. Schadensersatz
            ja/nein) zu ermöglichen.
          </p>

          <div className="spacer" />

          <p className="p">
            Die Anwendung richtet sich an Studierende und Interessierte im
            juristischen Bereich und dient als unterstützendes Analyse-Tool. Die
            Ergebnisse stellen keine Rechtsberatung dar, sondern eine technische
            Einschätzung auf Basis von Datenanalyse und Machine Learning.
          </p>

          <div className="spacer" />

          <h2 style={{ marginBottom: "8px" }}>Technischer Aufbau</h2>
          <p className="p">Das Projekt ist als moderne Web-App umgesetzt:</p>

          <ul style={{ marginTop: "8px", lineHeight: "1.6" }}>
            <li>Frontend: React (Vite)</li>
            <li>Backend: API zur Verarbeitung der Texte</li>
            <li>Analyse: Machine-Learning-Modell zur Klassifikation</li>
          </ul>

          <div className="spacer" />

          <h2 style={{ marginBottom: "8px" }}>Projektteam</h2>
          <p className="p">
            Dieses Projekt wurde im Rahmen einer universitären Gruppenarbeit
            entwickelt.
          </p>
        </div>
      </div>
    </div>
  );
}
