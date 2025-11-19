import React, { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'

function Box(props) {
    const meshRef = useRef()
    useFrame((state, delta) => (meshRef.current.rotation.x += delta))
    return (
        <mesh {...props} ref={meshRef}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial color={'orange'} />
        </mesh>
    )
}

export default function Viewer3D({ results }) {
    // TODO: Implement actual point cloud/mesh rendering from results
    // We need to fetch the aligned mesh or point cloud data from the backend
    // For now, just a placeholder scene

    return (
        <div className="viewer-container">
            <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <Box position={[0, 0, 0]} />
                <Grid infiniteGrid />
                <OrbitControls />
            </Canvas>
            <div className="viewer-overlay">
                <p>3D Viewer (Placeholder)</p>
                {results && <p>Loaded {results.length} results</p>}
            </div>
        </div>
    )
}
