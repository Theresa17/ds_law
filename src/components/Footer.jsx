export default function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-inner">
        <div className="footer-left">
          © {new Date().getFullYear()} <strong>VerdictIQ</strong>
        </div>

        <div className="footer-center">
        </div>

        <div className="footer-right">
        Datengestützte Fallanalyse
        </div>
      </div>
    </footer>
  );
}
