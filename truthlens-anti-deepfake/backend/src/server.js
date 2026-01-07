import "dotenv/config";
import express from "express";
import cors from "cors";
import connectDB from "./config/db.js";

import analyzeRoutes from "./routes/analyze.js";
import resultRoutes from "./routes/results.js";

const app = express();
const PORT = process.env.PORT || 5001;

connectDB();

app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});
app.use(cors());
app.use(express.json());
app.use("/uploads", express.static("uploads"));

app.use("/api/analyze", analyzeRoutes);
app.use("/api/results", resultRoutes);

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});