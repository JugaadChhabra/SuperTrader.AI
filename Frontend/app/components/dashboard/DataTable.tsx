import React from "react";
import Card from "../ui/Card";
import "./DataTable.css";

interface DataTableProps {
  columns: string[];
  data: Array<Record<string, any>>;
}

const DataTable: React.FC<DataTableProps> = ({ columns, data }) => (
  <Card className="data-table-card">
    <div className="data-table-wrapper">
      <table className="glass-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col}>{row[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </Card>
);

export default DataTable;
