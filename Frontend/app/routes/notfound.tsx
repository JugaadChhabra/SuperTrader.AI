import React from "react";
import Card from "../components/ui/Card";

const NotFound: React.FC = () => (
  <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
    <Card>
      <div style={{ textAlign: "center", color: "#fff" }}>
        <h1 style={{ fontSize: "3rem", marginBottom: "1rem" }}>404</h1>
        <p style={{ fontSize: "1.2rem" }}>Page Not Found</p>
      </div>
    </Card>
  </div>
);

export default NotFound;
