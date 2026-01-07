const generateSignatureCluster = (isLikelyAI, pointCount = 1500) => {
    const points = [];
    const scatterFactor = isLikelyAI ? 4.2 : 0.85;
    for (let i = 0; i < pointCount; i++) {
        const x = (Math.random() - 0.5) * scatterFactor;
        const y = (Math.random() - 0.5) * scatterFactor;
        const z = (Math.random() - 0.5) * scatterFactor;

        points.push(x, y, z);
    }
    return points;
};

module.exports = { generateSignatureCluster };
