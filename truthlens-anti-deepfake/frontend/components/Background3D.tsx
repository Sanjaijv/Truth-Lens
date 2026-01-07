'use client';

import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial, Float } from '@react-three/drei';
import * as THREE from 'three';

function ParticleField() {
    const count = 3000;
    const meshRef = useRef<THREE.Points>(null);

    const particles = useMemo(() => {
        const temp = new Float32Array(count * 3);
        const radius = 25;

        for (let i = 0; i < count; i++) {
            const theta = THREE.MathUtils.randFloatSpread(360);
            const phi = THREE.MathUtils.randFloatSpread(360);

            const x = radius * Math.sin(theta) * Math.cos(phi);
            const y = radius * Math.sin(theta) * Math.sin(phi);
            const z = radius * Math.cos(theta);

            temp.set([x, y, z], i * 3);
        }
        return temp;
    }, []);

    useFrame((state) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += 0.0001;
            meshRef.current.rotation.x += 0.00005;

            const x = (state.mouse.x * 0.1);
            const y = (state.mouse.y * 0.1);
            meshRef.current.rotation.y += x * 0.01;
            meshRef.current.rotation.x -= y * 0.01;
        }
    });

    return (
        <Points ref={meshRef} positions={particles} stride={3} frustumCulled={false}>
            <PointMaterial
                transparent
                color="#4f46e5"
                size={0.015}
                sizeAttenuation={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
                opacity={0.3}
            />
        </Points>
    );
}

function ForensicArtifact({ position, scale, rotationSpeed, type }: { position: [number, number, number], scale: number, rotationSpeed: number, type: 'wire' | 'solid' }) {
    const meshRef = useRef<THREE.Mesh>(null);

    useFrame((state, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * rotationSpeed;
            meshRef.current.rotation.z += delta * (rotationSpeed * 0.5);
        }
    });

    return (
        <Float speed={1.5} rotationIntensity={0.5} floatIntensity={0.5} position={position}>
            <mesh ref={meshRef} scale={scale}>
                {type === 'wire' ? (
                    <boxGeometry args={[1, 1, 1]} />
                ) : (
                    <octahedronGeometry args={[1, 0]} />
                )}
                <meshBasicMaterial
                    color="#4f46e5"
                    wireframe={type === 'wire'}
                    transparent
                    opacity={type === 'wire' ? 0.05 : 0.03}
                />
            </mesh>
        </Float>
    );
}

export default function Background3D() {
    return (
        <div className="fixed inset-0 -z-20 bg-slate-950 pointer-events-none">
            <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
                <ambientLight intensity={0.2} />
                <pointLight position={[10, 10, 10]} intensity={0.5} color="#4f46e5" />
                <ParticleField />
                <ForensicArtifact position={[-12, 5, -5]} scale={2.5} rotationSpeed={0.2} type="wire" />
                <ForensicArtifact position={[-10, -5, -8]} scale={1.8} rotationSpeed={0.15} type="solid" />
                <ForensicArtifact position={[12, 3, -4]} scale={2.2} rotationSpeed={0.1} type="solid" />
                <ForensicArtifact position={[11, -6, -6]} scale={2} rotationSpeed={0.25} type="wire" />
                <fog attach="fog" args={['#020617', 5, 35]} />
            </Canvas>
            <div className="absolute inset-0 bg-linear-to-b from-slate-950/60 via-transparent to-slate-950/80 pointer-events-none" />
        </div>
    );
}
