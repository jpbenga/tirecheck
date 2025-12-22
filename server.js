const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const PORT = 3000;

// --- CLASSE DE NORMALISATION FINALE ---
class Normalization extends tf.layers.Layer {
    constructor(config) {
        super(config || {});
        this.axis = config.axis || -1;
    }

    static get className() { return 'Normalization'; }

    // On crée les TROIS variables attendues par Keras 3
    build(inputShape) {
        const shape = [inputShape[inputShape.length - 1]];
        
        // La moyenne
        this.mean = this.addWeight('mean', shape, 'float32', tf.initializers.zeros(), null, false);
        // La variance
        this.variance = this.addWeight('variance', shape, 'float32', tf.initializers.ones(), null, false);
        // LE COMPTEUR (L'élément manquant qui causait ton erreur)
        this.count = this.addWeight('count', [], 'int32', tf.initializers.zeros(), null, false);
        
        this.built = true;
    }

    // Application de la formule mathématique de normalisation
    call(inputs) {
        return tf.tidy(() => {
            const x = inputs[0] || inputs;
            const epsilon = 1e-7;
            
            // Calcul : (x - mean) / sqrt(variance + epsilon)
            const mean = this.mean.read();
            const variance = this.variance.read();
            
            return tf.div(
                tf.sub(x, mean),
                tf.sqrt(tf.add(variance, tf.scalar(epsilon)))
            );
        });
    }

    getConfig() {
        const config = super.getConfig();
        return { ...config, axis: this.axis };
    }
}
tf.serialization.registerClass(Normalization);

// --- CONFIGURATION SERVEUR ---
const upload = multer({ dest: 'uploads/' });
app.use(express.static('public'));

let model;

async function loadModel() {
    try {
        console.log('⏳ Chargement du modèle avec synchronisation des poids...');
        const modelPath = path.join(__dirname, 'public', 'model_js', 'model.json');
        
        // Chargement du modèle (va maintenant trouver mean, variance ET count)
        model = await tf.loadLayersModel(`file://${modelPath}`);
        
        console.log('✅ MODÈLE CHARGÉ : Les poids sont correctement mappés.');
    } catch (err) {
        console.error('❌ ERREUR TECHNIQUE :', err.message);
    }
}
loadModel();

app.post('/analyze', upload.single('image'), async (req, res) => {
    if (!model) return res.status(503).json({ error: "Modèle non prêt" });
    if (!req.file) return res.status(400).json({ error: "Image manquante" });

    try {
        const imageBuffer = fs.readFileSync(req.file.path);

        const result = tf.tidy(() => {
            // Important : On décode et on redimensionne
            // Pas de division par 255 ici, car la couche Normalization s'en charge 
            // avec les vrais poids appris durant l'entraînement.
            const tensor = tf.node.decodeImage(imageBuffer, 3)
                .resizeBilinear([224, 224])
                .expandDims(0)
                .toFloat();
            
            return model.predict(tensor).dataSync();
        });

        // 0 = Defective, 1 = Good
        const classNames = ['Defective', 'Good'];
        const isDefective = result[0] > result[1];

        res.json({
            class: isDefective ? classNames[0] : classNames[1],
            confidence_defective: parseFloat(result[0]),
            confidence_good: parseFloat(result[1])
        });

    } catch (e) {
        console.error(e);
        res.status(500).json({ error: "Erreur lors de l'analyse" });
    } finally {
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    }
});

app.listen(PORT, () => console.log(`🚀 Serveur actif : http://localhost:${PORT}`));