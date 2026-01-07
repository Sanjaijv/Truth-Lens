// src/routes/results.js
import express from "express";
import { getResults } from "../controllers/resultsController.js";

const router = express.Router();

router.get("/", getResults);

export default router;
