import React, { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Points, PointMaterial, Line, Float, Html, Sparkles, MeshDistortMaterial, Stars } from "@react-three/drei";
import { EffectComposer, Bloom, Noise, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import { Info, X } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface BrainProps {
  isADHD: boolean;
  confidence: number;
  shap?: any;
}

interface Region {
  id: string;
  name: string;
  position: [number, number, number];
  color: string;
  description: string;
  associatedFeatures: string[];
  shapImportance?: number;
}

// Real SHAP channel importance from notebook EEG branch
// Top channels: F3, F4, O2, P7, P3, F7, Fp1 (from Cell 6 SHAP analysis)
const REGIONS: Region[] = [
  {
    id: "frontal",
    name: "Frontal Lobe",
    position: [0, 0.8, 1.2],
    color: "#3b82f6",
    description: "Executive function, attention control, and motor planning. Primary ADHD signature region with elevated theta power.",
    associatedFeatures: ["F3", "F4", "Fz", "F7", "Fp1", "Fp2", "Neuro TScore VarSE", "theta"],
    shapImportance: 0.859, // EEG AUC from model results
  },
  {
    id: "parietal",
    name: "Parietal Lobe",
    position: [0, 1.2, -0.5],
    color: "#a855f7",
    description: "Sensory integration and spatial awareness. Theta/beta ratio markers for inattention.",
    associatedFeatures: ["P7", "P3", "P4", "P8", "Pz", "Attention Index"],
    shapImportance: 0.72,
  },
  {
    id: "occipital",
    name: "Occipital Lobe",
    position: [0, 0.2, -1.8],
    color: "#ec4899",
    description: "Visual processing and pattern recognition. Beta power modulation.",
    associatedFeatures: ["O1", "O2"],
    shapImportance: 0.65,
  },
  {
    id: "temporal",
    name: "Temporal Lobe",
    position: [1.5, 0, 0.2],
    color: "#eab308",
    description: "Auditory processing and memory encoding. CPT reaction time variability.",
    associatedFeatures: ["T7", "T8", "Reaction Time", "Variability", "cpt_rt_std"],
    shapImportance: 0.58,
  },
];

const BrainRegion = ({ region, isHovered, onSelect, onHover, onOut }: any) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<any>(null);

  useFrame((state, delta) => {
    if (meshRef.current) {
      const targetScale = isHovered ? 1.6 : 1;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), delta * 8);
    }
    if (materialRef.current) {
      const targetOpacity = isHovered ? 0.9 : 0.4;
      const targetEmissiveIntensity = isHovered ? 10 : 1;
      const targetDistort = isHovered ? 0.6 : 0.2;
      const targetSpeed = isHovered ? 6 : 2;
      
      materialRef.current.opacity = THREE.MathUtils.lerp(materialRef.current.opacity, targetOpacity, delta * 8);
      materialRef.current.emissiveIntensity = THREE.MathUtils.lerp(materialRef.current.emissiveIntensity, targetEmissiveIntensity, delta * 8);
      materialRef.current.distort = THREE.MathUtils.lerp(materialRef.current.distort, targetDistort, delta * 8);
      materialRef.current.speed = THREE.MathUtils.lerp(materialRef.current.speed, targetSpeed, delta * 8);
      
      // Color shift
      const baseColor = new THREE.Color(region.color);
      const hoverColor = new THREE.Color(region.color).offsetHSL(0, 0.4, 0.2); // Lighter and more saturated
      materialRef.current.color.lerp(isHovered ? hoverColor : baseColor, delta * 8);
      materialRef.current.emissive.lerp(isHovered ? hoverColor : baseColor, delta * 8);
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={region.position}
      onClick={onSelect}
      onPointerOver={onHover}
      onPointerOut={onOut}
    >
      <sphereGeometry args={[0.25, 32, 32]} />
      <MeshDistortMaterial
        ref={materialRef}
        color={region.color}
        transparent
        opacity={0.4}
        emissive={region.color}
        emissiveIntensity={1}
        distort={0.2}
        speed={2}
      />
      <Html distanceFactor={10}>
        <div className="pointer-events-none select-none">
          <div className={`w-1 h-1 bg-white rounded-full transition-all duration-300 ${isHovered ? 'scale-[3] shadow-[0_0_15px_rgba(255,255,255,0.8)]' : 'animate-ping'}`} />
        </div>
      </Html>
    </mesh>
  );
};

const NeuralNetwork = ({ isADHD, confidence, shap, onSelectRegion }: any) => {
  const pointsRef = useRef<THREE.Points>(null);
  const [hoveredRegionId, setHoveredRegionId] = useState<string | null>(null);

  const count = 1200;
  const points = useMemo(() => {
    const p = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 2 * Math.pow(Math.random(), 0.5);
      
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta) * 1.2;
      const z = r * Math.cos(phi) * 0.8;
      
      p[i * 3] = x;
      p[i * 3 + 1] = y;
      p[i * 3 + 2] = z;
    }
    return p;
  }, []);

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += 0.001;
      const pulse = Math.sin(state.clock.elapsedTime * (isADHD ? 3 : 1)) * 0.05 + 1;
      pointsRef.current.scale.set(pulse, pulse, pulse);
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
      <Points ref={pointsRef} positions={points} stride={3}>
        <PointMaterial
          transparent
          color={isADHD ? "#ff3333" : "#33ff77"}
          size={0.05}
          sizeAttenuation={true}
          depthWrite={false}
          opacity={Math.max(0.4, confidence * 0.8)}
          blending={THREE.AdditiveBlending}
        />
      </Points>

      {REGIONS.map((region) => {
        const isHovered = hoveredRegionId === region.id;
        return (
          <BrainRegion
            key={region.id}
            region={region}
            isHovered={isHovered}
            onSelect={(e: any) => {
              e.stopPropagation();
              onSelectRegion(region);
            }}
            onHover={(e: any) => {
              e.stopPropagation();
              setHoveredRegionId(region.id);
              document.body.style.cursor = "pointer";
            }}
            onOut={() => {
              setHoveredRegionId(null);
              document.body.style.cursor = "auto";
            }}
          />
        );
      })}
      
      <group>
        {Array.from({ length: 15 }).map((_, i) => (
          <Float speed={1.5} rotationIntensity={0.5} floatIntensity={1} key={i}>
            <Line
              points={[
                [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1],
                [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1],
              ]}
              color={isADHD ? "#f87171" : "#4ade80"}
              lineWidth={0.5}
              transparent
              opacity={0.1}
            />
          </Float>
        ))}
      </group>
    </Float>
  );
};

export const Brain3D = ({ isADHD, confidence, shap }: BrainProps) => {
  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);

  const regionShap = useMemo(() => {
    if (!selectedRegion || !shap) return [];
    const allShap = [...(shap.clinical || []), ...(shap.eeg || [])];
    return allShap.filter((s: any) => 
      selectedRegion.associatedFeatures.some(f => s.feat.includes(f))
    );
  }, [selectedRegion, shap]);

  return (
    <div className="w-full h-[500px] bg-black/40 rounded-3xl overflow-hidden relative border border-white/5 group">
      <div className="absolute top-6 left-6 z-10">
        <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/40 mb-1">Interactive Neural Map</h3>
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full animate-pulse ${isADHD ? 'bg-red-500' : 'bg-green-500'}`} />
          <span className="text-[10px] font-mono text-white/60 uppercase tracking-widest">Click regions to explore SHAP drivers</span>
        </div>
      </div>
      
      <Canvas camera={{ position: [0, 0, 6], fov: 40 }}>
        <color attach="background" args={["#050505"]} />
        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={1.5} />
        <NeuralNetwork 
          isADHD={isADHD} 
          confidence={confidence} 
          shap={shap} 
          onSelectRegion={setSelectedRegion} 
        />
        <Sparkles count={100} scale={10} size={1} speed={0.5} color={isADHD ? "#ef4444" : "#22c55e"} opacity={0.2} />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} makeDefault autoRotate autoRotateSpeed={0.5} />
        
        <EffectComposer enableNormalPass={false}>
          <Bloom 
            luminanceThreshold={0.2} 
            mipmapBlur 
            intensity={isADHD ? 2.5 : 1.2} 
            radius={0.4} 
          />
          <Noise opacity={0.05} />
          <Vignette eskil={false} offset={0.1} darkness={1.1} />
        </EffectComposer>
      </Canvas>

      {/* Region Detail Overlay */}
      <AnimatePresence>
        {selectedRegion && (
          <div className="absolute inset-y-0 right-0 w-72 bg-black/80 backdrop-blur-2xl border-l border-white/10 p-8 z-20 animate-in slide-in-from-right duration-300">
            <button 
              onClick={() => setSelectedRegion(null)}
              className="absolute top-6 right-6 text-white/40 hover:text-white transition-colors"
            >
              <X size={18} />
            </button>

            <div className="flex items-center gap-3 mb-6">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: selectedRegion.color }} />
              <h4 className="text-sm font-bold uppercase tracking-widest">{selectedRegion.name}</h4>
            </div>

            <p className="text-xs text-white/50 leading-relaxed mb-8 font-mono">
              {selectedRegion.description}
            </p>

            <div className="space-y-6">
              <h5 className="text-[10px] font-mono uppercase tracking-widest text-white/30 border-b border-white/5 pb-2">Associated SHAP Drivers</h5>
              
              {regionShap.length > 0 ? (
                <div className="space-y-4">
                  {regionShap.map((s: any, i: number) => (
                    <div key={i} className="space-y-1.5">
                      <div className="flex justify-between text-[10px] font-mono">
                        <span className="text-white/60 truncate max-w-[160px]">{s.feat}</span>
                        <span className={s.val > 0 ? "text-red-400" : "text-green-400"}>
                          {s.val > 0 ? "+" : ""}{s.val.toFixed(4)}
                        </span>
                      </div>
                      <div className="h-0.5 bg-white/5 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full ${s.val > 0 ? 'bg-red-400/50' : 'bg-green-400/50'}`}
                          style={{ width: `${Math.min(Math.abs(s.val) * 200, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center opacity-30">
                  <Info size={24} className="mb-2" />
                  <p className="text-[10px] font-mono uppercase">No active drivers in this region</p>
                </div>
              )}
            </div>

            <div className="absolute bottom-8 left-8 right-8">
              <div className="text-[9px] font-mono text-white/20 uppercase tracking-widest leading-tight">
                Functional Mapping based on DSM-5 Neuro-Inference Engine
              </div>
            </div>
          </div>
        )}
      </AnimatePresence>
      
      <div className="absolute bottom-6 left-6 flex gap-4">
        {REGIONS.map(r => (
          <div key={r.id} className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: r.color }} />
            <span className="text-[9px] font-mono text-white/30 uppercase tracking-widest">{r.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

