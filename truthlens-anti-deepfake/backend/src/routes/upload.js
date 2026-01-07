import express from "express";
import multer from "multer";
import path from "path";
import { quickScan } from "./analyze.js";

const router = express.Router();

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "uploads/");
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    },
});

const upload = multer({ storage });

router.post("/", upload.single("video"), quickScan);

export default router;
