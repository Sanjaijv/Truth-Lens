import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './components/ui/card';
import { ShieldCheck, Video, Activity, AlertTriangle } from 'lucide-react';
import './popup.css';

const App = () => {
    const [status, setStatus] = useState("Click to scan page for faces");
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    const handleAnalyze = async () => {
        // Query active tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab) return;

        // Show immediate feedback in popup before it closes
        setStatus("Launching TruthLens...");

        try {
            // 1. Inject CSS (Important for overlay visibility)
            await chrome.scripting.insertCSS({
                target: { tabId: tab.id },
                files: ['dist/content.css']
            }).catch((e) => console.log("CSS injection skipped:", e));

            // 2. Inject JS (Important for logic)
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ['dist/content.js']
            }).catch((e) => console.log("Script injection skipped:", e));

            console.log("Transmission: Sending ANALYZE_VIDEO to tab", tab.id);

            // 3. Send Message and Wait for dispatch
            // We use a Promise wrapper to ensure we wait for the message to leave the popup context
            // before we kill the popup with window.close()

            await new Promise((resolve) => {
                setTimeout(() => {
                    chrome.tabs.sendMessage(tab.id, { action: "ANALYZE_VIDEO" })
                        .finally(() => {
                            console.log("Transmission: Processing complete. Closing popup.");
                            resolve();
                        });
                }, 100);
            });

            window.close();

        } catch (error) {
            console.error("Popup Fire error:", error);
            setStatus("Error: " + error.message);
        }
    };

    return (
        <div className="p-4 w-[320px] h-[400px] flex items-center justify-center bg-background text-foreground font-sans">
            <Card className="w-full h-full border-border bg-card/50 backdrop-blur-sm flex flex-col justify-between shadow-2xl rounded-[2rem]">
                <CardHeader className="pb-2">
                    <div className="flex items-center gap-3 mb-1">
                        <div className="p-2 bg-primary/10 rounded-2xl">
                            <ShieldCheck className="h-6 w-6 text-primary" />
                        </div>
                        <div>
                            <CardTitle className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-blue-500">
                                TruthLens
                            </CardTitle>
                            <CardDescription className="text-muted-foreground text-xs font-medium">
                                Anti-Deepfake Verification
                            </CardDescription>
                        </div>
                    </div>
                </CardHeader>
                <CardContent className="space-y-4 flex-1 flex flex-col justify-center">
                    <div className={`
                        flex items-center justify-center p-4 rounded-2xl border
                        min-h-[100px] transition-colors duration-300
                        ${status.includes("Error")
                            ? "bg-red-500/10 border-red-500/20 text-red-200"
                            : "bg-secondary/50 border-border text-muted-foreground"}
                    `}>
                        <p className="text-sm text-center font-medium leading-relaxed">
                            {status}
                        </p>
                    </div>
                </CardContent>
                <CardFooter className="pt-2">
                    <Button
                        onClick={handleAnalyze}
                        disabled={isAnalyzing}
                        className="w-full h-11 text-base font-medium transition-all duration-300 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 border-none text-white shadow-lg shadow-purple-500/20 rounded-full"
                    >
                        {isAnalyzing ? (
                            <>
                                <Activity className="mr-2 h-5 w-5 animate-spin" />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <Video className="mr-2 h-5 w-5" />
                                Analyze Video
                            </>
                        )}
                    </Button>
                </CardFooter>
            </Card>
        </div>
    );
};

const root = createRoot(document.getElementById('root'));
root.render(<App />);
