// src/controllers/resultsController.js
import ScanResult from "../models/ScanResult.js";

export const getResults = async (req, res) => {
  try {
    const results = await ScanResult.find().sort({ createdAt: -1 });
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch results" });
  }
};
