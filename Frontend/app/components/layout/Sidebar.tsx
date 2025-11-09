import React from "react";
import "./Sidebar.css";

const Sidebar: React.FC = () => (
  <aside className="glass-sidebar">
    <nav>
      <ul>
        <li><a href="/dashboard">Dashboard</a></li>
        <li><a href="/profile">Profile</a></li>
        <li><a href="/logout">Logout</a></li>
      </ul>
    </nav>
  </aside>
);

export default Sidebar;
