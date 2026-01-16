import { NavLink } from "react-router-dom";
import LogoDS from "../assets/Logo_DS.png";

export default function Header() {
  const cls = ({ isActive }) => (isActive ? "active" : "");

  return (
    <header className="header">
      <div className="container header-inner">
        <div className="brand">
          <img
            src={LogoDS}
            alt="DS Law"
            className="brand-logo"
          />

          <div className="brand-title">
            <div>VerdictIQ</div>
            <small>KI-gestützte Urteilsanalyse</small>
          </div>
        </div>

        <nav className="nav">
          <NavLink to="/" className={cls}>Home</NavLink>
          <NavLink to="/history" className={cls}>Verlauf</NavLink>
          <NavLink to="/about" className={cls}>Über uns</NavLink>
        </nav>
      </div>
    </header>
  );
}

