'use client';

import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

function PointCloud({ isLikelyAI }: { isLikelyAI: boolean }) {
    const count = 1500;
    const meshRef = useRef<THREE.Points>(null);

    const particles = useMemo(() => {
        const temp = new Float32Array(count * 3);
        // Use a simple deterministic sequence to avoid Math.random() lint error
        for (let i = 0; i < count; i++) {
            const factor = isLikelyAI ? 4 : 0.8;
            const x = ((Math.sin(i * 12.9898) * 43758.5453) % 1) * factor;
            const y = ((Math.sin(i * 78.233) * 43758.5453) % 1) * factor;
            const z = ((Math.sin(i * 45.164) * 43758.5453) % 1) * factor;
            temp.set([x, y, z], i * 3);
        }
        return temp;
    }, [isLikelyAI]);

    useFrame(() => {
        if (meshRef.current) {
            meshRef.current.rotation.y += 0.002;
            meshRef.current.rotation.z += 0.001;
        }
    });

    return (
        <Points ref={meshRef} positions={particles} stride={3} frustumCulled={false}>
            <PointMaterial
                transparent
                color={isLikelyAI ? "#f43f5e" : "#4f46e5"}
                size={0.05}
                sizeAttenuation={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
            />
        </Points>
    );
}

export default function SignatureCube({ isLikelyAI }: { isLikelyAI: boolean }) {
    return (
        <div className="w-full h-[400px] bg-slate-950 rounded-[48px] overflow-hidden relative border border-slate-800 shadow-2xl">
            <div className="absolute top-8 left-8 z-10 font-mono text-[10px] text-indigo-400 uppercase tracking-[0.3em] space-y-2">
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                    3D Signature Map v2.4
                </div>
                <div className="text-slate-500">Coordinate: PLOT_XYZ_FORENSIC</div>
            </div>
            <div className="absolute bottom-8 right-8 z-10 font-mono text-[10px] text-slate-500 uppercase tracking-widest">
                {isLikelyAI ? "[ SIGNAL_SCATTER_DETECTED ]" : "[ STABLE_CLUSTER_VERIFIED ]"}
            </div>
            <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <PointCloud isLikelyAI={isLikelyAI} />
                <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
                <mesh>
                    <boxGeometry args={[4, 4, 4]} />
                    <meshBasicMaterial color="#1e293b" wireframe transparent opacity={0.2} />
                </mesh>
            </Canvas>
        </div>
    );
}