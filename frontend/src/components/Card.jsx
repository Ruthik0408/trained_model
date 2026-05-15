import React from "react";
export default function Card({ title, children }) {
    return (<div style={{ background: "#fff", borderRadius: 16, padding: 16, boxShadow: "0 4px 16px rgba(0,0,0,0.06)" }}>
      {title ? <h3 style={{ margin: "0 0 12px" }}>{title}</h3> : null}
      {children}
    </div>);
}
