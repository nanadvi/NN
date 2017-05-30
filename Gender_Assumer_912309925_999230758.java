import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
public class Gender_Assumer_912309925_999230758 {

    // array of test images
    public static ArrayList<Image> tests = new ArrayList<>();

    // array of training images
    public static ArrayList<Image> train = new ArrayList<>();


    // splitting the training array into sets
    public static ArrayList<Image> set1 = new ArrayList<>();
    public static ArrayList<Image> set2 = new ArrayList<>();
    public static ArrayList<Image> set3 = new ArrayList<>();
    public static ArrayList<Image> set4 = new ArrayList<>();
    public static ArrayList<Image> set5 = new ArrayList<>();

    static int[][] doctored = {
            {0,1,2,3,4,5}
    };

    // Layer class; this represents the input layer
    public static class Layer {
        double[] pixels;    // array of ~15k pixels
        int gender;             // male = 1
                                // female = 2

        public Layer(int[][] pixels, int gender) {
            this.pixels = condense(pixels);
            this.gender = gender;
        }

        public double[] condense(int[][] pixels) {
            int k = 0;
            double[] result = new double[pixels.length*pixels[0].length];
            for (int i = 0; i < pixels.length; i++) {
                for (int j = 0; j < pixels[i].length; j++) {
                    result[k] = pixels[i][j];
                    k++;
                }
            }

            return result;
        }
    }

    // this neuron class is a node in the hidden layer
    public static class Neuron {
        Layer previous; // previous layer
        Neuron[] prevN;   // previous neuron
        double[] previousWeights;
        double summedValue; // multiply each pixel with weight and this is the sum of all those numbers
        double postSigmoid;
        double weight;
        double delta;
        boolean firstHidden = false;

        // construct as a hidden layer neuron (first hidden layer)
        public Neuron(Layer layer) {
            this.previous = layer;
            this.previousWeights = setLayerWeights(this.previous.pixels.length);
            this.summedValue = setHold(this.previous);
            this.postSigmoid = calculateSigmoid(this.summedValue);
            this.weight = Math.random();
            this.firstHidden = true;
        }


        // construct as a hidden layer neuron (not first hidden layer)
        public Neuron(Neuron[] neuron) {
            this.prevN = neuron;
            this.weight = Math.random();
            for (int i = 0; i < neuron.length; i++) {
                this.summedValue += (neuron[i].summedValue * neuron[i].weight);
            }
            this.postSigmoid = calculateSigmoid(this.summedValue);
        }

        public void changeLayer(Layer layer) {
            this.previous = layer;
            this.summedValue = setHold(this.previous);
            this.postSigmoid = calculateSigmoid(this.summedValue);
            // this.weight = Math.random();
            this.firstHidden = true;
        }

        public void updateHidden() {
            for (int i = 0; i < this.prevN.length; i++) {
                this.summedValue += this.prevN[i].summedValue;
            }
            this.summedValue *= this.weight;
            this.postSigmoid = calculateSigmoid(this.summedValue);
        }

        public double setHold(Layer layer) {
            double result = 0.0;
            for (int i = 0; i < layer.pixels.length; i++) {
                result += layer.pixels[i] * this.previousWeights[i];
            }

            return result;
        }

        public double[] setLayerWeights(int size) {
            double[] result = new double[size];

            for (int i = 0; i < size; i++) {
                result[i] = Math.random() / 2500000;
            }

            return result;
        }

        public double calculateSigmoid(double num) {
            double result = 0.0;
            result = 1.0 / (1.0 + Math.exp(-1 * num));
            return result;
        }

        public void setDelta(double deltaK) {
            double result = 0.0;
            result = (this.postSigmoid) * (1 - this.postSigmoid) * (deltaK) * (this.weight);
            this.delta = result;
        }

        public void changeWeightArray() {
            for (int i = 0; i < this.previousWeights.length; i++) {
                this.previousWeights[i] = this.previousWeights[i] + (this.delta * this.previousWeights[i] * -0.2);
            }
        }

        public void changeWeights(double delta) {
            this.weight += (delta * this.weight * -0.2);
            if (this.firstHidden) {
                changeWeightArray();
            }
            else {
                for (int i = 0; i < prevN.length; i++) {
                    this.prevN[i].changeWeights(delta);
                }
            }
        }
    }

    public static class OutputNeuron {
        Neuron[] previousHiddenLayer;
        double outputValue;
        double errorValue;
        double delta;

        // construct an output layer neuron
        public OutputNeuron(Neuron[] neurons) {
            this.previousHiddenLayer = neurons;
            this.outputValue = setOutputNode(neurons);
        }

        public void updateOutputs() {
            for (int i = 0; i < previousHiddenLayer.length; i++) {
                if (!previousHiddenLayer[i].firstHidden) {
                    previousHiddenLayer[i].updateHidden();
                }
            }
            this.outputValue = setOutputNode(this.previousHiddenLayer);
        }

        public double setOutputNode(Neuron[] neurons) {
            double result1 = 0.0;
            double result2 = 0.0;
            for (int i = 0; i < neurons.length; i++) {
                result1 += neurons[i].postSigmoid * neurons[i].weight;
            }
            result2 = calculateSigmoid(result1);
            return result2;
        }

        public double calculateSigmoid(double num) {
            double result = 0.0;
            result = 1.0 / (1.0 + Math.exp(-1 * num));
            return result ;
        }

        public void setErrorValue(double expectedValue) {
            double expected = expectedValue;

            this.errorValue = this.outputValue - expected;
        }

        public void updateValue(double expectedValue) {
            setErrorValue(expectedValue);
            setDelta(expectedValue);
            for (int i = 0; i < this.previousHiddenLayer.length; i++) {
                this.previousHiddenLayer[i].setDelta(this.delta);
                this.previousHiddenLayer[i].changeWeights(this.delta);
            }
            this.outputValue = setOutputNode(this.previousHiddenLayer);
        }

        public void setDelta(double expectedValue) {
            double result = 0.0;
            result = (this.outputValue) * (1 - this.outputValue) * (this.errorValue);
            this.delta = result;
        }

    }

    public static void main(String[] args) throws FileNotFoundException {
        File male = new File(args[1]);
        File female = new File(args[2]);
        File test = new File(args[4]);
        loadImage(male, "M");
        loadImage(female, "F");
        loadImage(test, "T");
        splitTrainingSet();
        //train("1_1_1.txt");

        // construct input layer
        ArrayList<Layer> inputLayer = new ArrayList<Layer>();
        // test(inputLayer);
        System.out.println("Cross Validation");
        testFive(inputLayer, 5);
        inputLayer.clear();
        testFive(inputLayer, 4);
        inputLayer.clear();
        testFive(inputLayer, 3);
        inputLayer.clear();
        testFive(inputLayer, 2);
        inputLayer.clear();
        testFive(inputLayer, 1);
        inputLayer.clear();

        for (int i = 0; i < train.size(); i++) {
            inputLayer.add(new Layer(train.get(i).greyscale, train.get(i).gender));
        }
        System.out.println("General");
        test(inputLayer, tests);
    }

    public static void testFive(ArrayList<Layer> inputLayer, int testingIndex) {
        for (int q = 0; q < 5; q++) {
            if (testingIndex != 1) {
                for (int i = 0; i < set1.size(); i++) {
                    inputLayer.add(new Layer(set1.get(i).greyscale, set1.get(i).gender));
                }
            } else if (testingIndex != 2) {
                for (int i = 0; i < set2.size(); i++) {
                    inputLayer.add(new Layer(set2.get(i).greyscale, set2.get(i).gender));
                }
            } else if (testingIndex != 3) {
                for (int i = 0; i < set3.size(); i++) {
                    inputLayer.add(new Layer(set3.get(i).greyscale, set3.get(i).gender));
                }
            } else if (testingIndex != 4) {
                for (int i = 0; i < set4.size(); i++) {
                    inputLayer.add(new Layer(set4.get(i).greyscale, set4.get(i).gender));
                }
            } else if (testingIndex != 5) {
                for (int i = 0; i < set5.size(); i++) {
                    inputLayer.add(new Layer(set5.get(i).greyscale, set5.get(i).gender));
                }
            }
        }

        if (testingIndex == 1 ) {
            test(inputLayer, set1);
        }
        else if (testingIndex == 2 ) {
            test(inputLayer, set2);
        }
        else if (testingIndex == 3 ) {
            test(inputLayer, set3);
        }
        else if (testingIndex == 4 ) {
            test(inputLayer, set4);
        }
        else if (testingIndex == 5 ) {
            test(inputLayer, set5);
        }
    }
    // create neural network
    public static void construct(ArrayList<Neuron[]> hiddenLayers, Layer firstPicture, int numberOfLayers, int numberOfNodes) {
        // create first hidden layer
        int index = 0;
        int save = numberOfLayers;
        Neuron[] firstHidden = new Neuron[numberOfNodes];
        for (int i = 0; i < numberOfNodes; i++) {
            firstHidden[i] = new Neuron(firstPicture);
        }
        hiddenLayers.add(firstHidden);
        save--;
        while (save != 0) {
            Neuron[] newHidden = new Neuron[numberOfNodes];
            for (int i = 0; i < numberOfNodes; i++) {
                newHidden[i] = new Neuron(hiddenLayers.get(index));
            }
            hiddenLayers.add(newHidden);
            index++;
            save--;
        }
    }

    public static void test(ArrayList<Layer> inputLayer, ArrayList<Image> testing) {
        // construct network
        ArrayList<Neuron[]> hiddenLayers = new ArrayList<Neuron[]>();
        if (hiddenLayers.isEmpty()) {
            construct(hiddenLayers, inputLayer.get(0), 2, 5);
        }

        OutputNeuron output = new OutputNeuron(hiddenLayers.get(hiddenLayers.size()-1));
        // System.out.println("Output: " + output.outputValue + " Delta: " + output.delta + " Error: " + output.errorValue);

        train(inputLayer, hiddenLayers, output);
        // train(inputLayer, hiddenLayers, output);
        // train(inputLayer, hiddenLayers, output);

        // System.out.println("TESTING HERE");

        ArrayList<Neuron[]> clone = new ArrayList<Neuron[]>();
        for (Neuron[] n : hiddenLayers) {
            clone.add(n.clone());
        }
        int men = 0;
        int fem = 0;
        for (int i = 0; i < testing.size(); i++) {
            Layer lay = new Layer(testing.get(i).greyscale, testing.get(i).gender);
            for (int k = 0; k < hiddenLayers.get(0).length; k++) {
                clone.get(0)[k].changeLayer(lay);
            }
            OutputNeuron putout = new OutputNeuron(clone.get(clone.size()-1));
            double oldOutput = putout.outputValue;
            putout.updateOutputs();
            if (putout.outputValue < oldOutput) {
                System.out.print("Is Male   |  ");
                men++;
            } else {
                System.out.print("Is Female |  ");
                fem++;
            }
            System.out.println("Testing File Output: " + putout.outputValue);

        }
        System.out.println("Amount of Male: " + men + " Amount of Female: " + fem);
    }

    // do the actual training
    public static void train(ArrayList<Layer> inputLayer, ArrayList<Neuron[]> hiddenLayers, OutputNeuron output) {
        for (int i = 1; i < inputLayer.size(); i++) {
            if (inputLayer.get(i).gender == 0) {
                // System.out.print("Gender: Male ");
                output.updateValue(0.5);
            }
            else if (inputLayer.get(i).gender == 1) {
                // System.out.print("Gender: Female ");
                output.updateValue(1.0);
            }

            // System.out.println("Output: " + output.outputValue + " Delta: " + output.delta + " Error: " + output.errorValue);

            for (int k = 0; k < hiddenLayers.get(0).length; k++) {
                hiddenLayers.get(0)[k].changeLayer(inputLayer.get(i));
            }
        }
    }

    // splits the large training data into 5 smaller ones
    public static void splitTrainingSet(){
        long seed = System.nanoTime();
        int trainingLen = train.size();
        Collections.shuffle(train, new Random(seed));

        // after shuffling the list, split them into 5 separate lists
        for (int i = 0; i < trainingLen; i++) {
            if (i % 5 == 0) {
                set1.add(train.get(i));
            }
            else if (i % 5 == 1) {
                set2.add(train.get(i));
            }
            else if (i % 5 == 2) {
                set3.add(train.get(i));
            }
            else if (i % 5 == 3) {
                set4.add(train.get(i));
            }
            else if (i % 5 == 4) {
                set5.add(train.get(i));
            }
        }

    }

    // read the .txt file
    public static void loadImage(File dir, String s) throws FileNotFoundException {

        File[] listOfFiles = dir.listFiles();

        if(listOfFiles != null) {
            for (File fileEntry : listOfFiles) {
                int[][] temp = pixelArray(dir.getName() + "/" + fileEntry.getName());
                if(s.equals("M"))
                    train.add(new Image(temp, 0, fileEntry.getName()));
                else if(s.equals("F"))
                    train.add(new Image(temp, 1, fileEntry.getName()));
                else
                    tests.add(new Image(temp, 2, fileEntry.getName()));
            }
        }
    }

    // image class; holds 2d pixel array, gender, and filename
    public static class Image {
        int[][] greyscale;
        int gender; // 1 if male  2 is female
        String fileName;

        public Image(int[][] greys, int gender, String file) {
            this.greyscale = greys;
            this.gender = gender;
            this.fileName = file;
        }
    }

    // turns the array of the given files to actual 128x120 2d arrays
    public static int[][] pixelArray(String filename) throws FileNotFoundException {
        int[][] pixels = new int[120][128];
        String take = "";
        File inFile = new File(filename);
        Scanner reader = new Scanner(inFile);
        int i = 0;
        int j = 0;
        while (reader.hasNextInt()) {
            if (j == 128) {
                i++;
                j = 0;
            }
            pixels[i][j] = reader.nextInt();
            j++;
        }
        return pixels;
    }

    // prints a 2d double array
    public static void print2DArrayDouble(double[][] arr) throws FileNotFoundException {
        //FileOutputStream f = new FileOutputStream("log_file.txt");
        //System.setOut(new PrintStream(f));
        for(int i = 0; i < arr.length; ++i) {
            for(int j = 0; j < arr[i].length; ++j) {
                System.out.print(arr[i][j] + " ");
            }

            System.out.println();
        }
    }

    public static void printArrayDouble(double[] arr) throws FileNotFoundException {
        System.out.println("Size of array: " + arr.length);
        for (int i = 0; i < 10; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }


    // prints a 2d int array
    public static void print2DArrayInt(int[][] arr) throws FileNotFoundException {
        //FileOutputStream f = new FileOutputStream("log_file.txt");
        //System.setOut(new PrintStream(f));
        for(int i = 0; i < arr.length; ++i) {
            for(int j = 0; j < arr[i].length; ++j) {
                System.out.print(arr[i][j] + " ");
            }

            System.out.println();
        }
    }
}
