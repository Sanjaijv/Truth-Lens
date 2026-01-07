// src/routes/analyze.js
import express from "express";
import upload from "../utils/uploadHandler.js";
import { analyzeVideo } from "../controllers/analyzeController.js";

const router = express.Router();

router.post("/", upload.single("video"), analyzeVideo);

export default router;
