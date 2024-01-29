import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

// I didn't cheat!
public class NaiveBayesClassifier {
    // Data structures to store word counts and probabilities for spam & ham classes
    private final Map<String, Integer> spamWordCounts = new HashMap<>(); // Map with spam word -> word count
    private final Map<String, Integer> hamWordCounts = new HashMap<>(); // Map with ham word -> word count
    private double spamPriorProb; // Prior prob for spam class
    private double hamPriorProb; // Prior prob for ham class
    private final Set<String> vocab = new HashSet<>(); // Set to store unique words in the vocab
    private static final boolean debug = false; // Debugging constant
    private static Double totalNumEmails; // Spam emails + ham emails
    private int totalNumSpam; // Num of spam emails
    private int totalNumHam; // Num of ham emails

    // Method to train classifier
    public void train(String spamTrainingFile, String hamTrainingFile) {
        // Read and process training data to build vocab and calc word probs
        // Calc prior probs for spam and ham classes
        ArrayList<String> spamEmails = parseData(spamTrainingFile);
        ArrayList<String> hamEmails = parseData(hamTrainingFile);

        ArrayList<String> bodySpamEmails = parseOnlyBodyData(spamTrainingFile);
        totalNumSpam = bodySpamEmails.size();
        ArrayList<String> bodyHamEmails = parseOnlyBodyData(hamTrainingFile);
        totalNumHam = bodyHamEmails.size();

        totalNumEmails = (double) (bodySpamEmails.size() + bodyHamEmails.size());

        // Calc prior probs
        spamPriorProb = (double) bodySpamEmails.size() / (bodySpamEmails.size() + bodyHamEmails.size());
        hamPriorProb = 1 - spamPriorProb;

        // Build vocab from both emails
        buildVocab(bodySpamEmails, bodyHamEmails);

        // Calc word counts for words in both classes
        calcWordCounts(spamEmails, hamEmails);


        // Debugging info
        if (debug) {
            System.out.println("Training from " + spamTrainingFile + " and " + hamTrainingFile);
            System.out.println("TRAIN: Num of emails " + bodySpamEmails.size() + " vs " + bodyHamEmails.size());
            System.out.println("Total num of emails: " + totalNumEmails);
            System.out.println("Entire vocab " + vocab);
            System.out.println("Entire vocab size is " + vocab.size());
            System.out.println("Spam words " + spamWordCounts);
            System.out.println("Ham words " + hamWordCounts);
        }
    }


    // Method to build vocab form spam and ham emails
    public void buildVocab(ArrayList<String> spamEmails, ArrayList<String> hamEmails) {
        for (String email : spamEmails) {
            //System.out.println("SPAM Analyzing email: " + email);
            String[] words = email.split("\\s+");
            //System.out.println("SPAM Words from email: " + email + " are: " + Arrays.toString(words));
            Set<String> uniqueWords = new HashSet<>(Arrays.asList(words)); // Use a set to track unique words
            //System.out.println("SPAM Unique words from words: " + Arrays.toString(words) + " are: " + uniqueWords);
            //System.out.println("SPAM Looking at word: " + word + " in unique words= " + uniqueWords);
            vocab.addAll(uniqueWords);
        }

        for (String email : hamEmails) {
            String[] words = email.split("\\s+");
            Set<String> uniqueWords = new HashSet<>(Arrays.asList(words));
            vocab.addAll(uniqueWords);

        }
        System.out.println("VOCAB: " + vocab.size());
    }

    // Method to calc word probs for a given class
    public double[] calcProbs(String testEmail) {
        int trueFeatures = 0;
        double spamProb = (totalNumSpam / totalNumEmails);
        double hamProb = (totalNumHam / totalNumEmails);
        double totalSpamProb = Math.log(spamProb);
        double totalHamProb = Math.log(hamProb);
        String[] emailWords = testEmail.split("\\s+");
        Set<String> testWords = new HashSet<>(Arrays.asList(emailWords));

        for (String word : vocab) {
            if (testWords.contains(word)) {
                trueFeatures++;
                double wordSpamProb = getWordProb(spamWordCounts, word, totalNumSpam);
                double wordHamProb = getWordProb(hamWordCounts, word, totalNumHam);

                totalSpamProb += Math.log(wordSpamProb);
                totalHamProb += Math.log(wordHamProb);
            } else {
                double wordSpamProb = 1.0 - getWordProb(spamWordCounts, word, totalNumSpam);
                double wordHamProb = 1.0 - getWordProb(hamWordCounts, word, totalNumHam);

                totalSpamProb += Math.log(wordSpamProb);
                totalHamProb += Math.log(wordHamProb);
            }
        }
        return new double[]{totalSpamProb, totalHamProb, trueFeatures};
    }

    // Helper function for calcProbs
    private double getWordProb(Map<String, Integer> wordCounts, String word, int totalClassCount) {
        int count = wordCounts.getOrDefault(word, 0);
        return (count + 1.0) / (totalClassCount + 2.0);
    }

    // Calc word counts for spam and ham emails
    public void calcWordCounts(ArrayList<String> spamEmails, ArrayList<String> hamEmails) {
        for (String email : spamEmails) {
            updateWordCounts(email, spamWordCounts);
        }

        for (String email : hamEmails) {
            updateWordCounts(email, hamWordCounts);
        }
    }

    // Helper function for calcWordCounts to update the count
    private void updateWordCounts(String email, Map<String, Integer> wordCounts) {
        String[] words = email.split("\\s+");
        Set<String> uniqueWords = new HashSet<>(Arrays.asList(words));

        for (String word : uniqueWords) {
            if (!word.isEmpty()) {
                // Add non-empty words to vocab and update word counts
                wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
            }
        }
    }

    // Method to test classifier on spam and ham data
    public int[] test(String TestingFile, int isSpam) {
        String label;
        int correctLabeled = 0;

        if (debug) {
            System.out.println("Beginning tests.");
            System.out.println("Testing from " + TestingFile + " and " + TestingFile);
        }

        ArrayList<String> testEmails = parseOnlyBodyData(TestingFile);

        //Testing:
        for (int i = 0; i < testEmails.size(); i++) {
            double[] probs = calcProbs(testEmails.get(i));
            if (probs[0] > probs[1]) label = "spam";
            else label = "ham";
            String s = String.format("Test %d %d/%d features true %.3f %.3f %s",
                    i + 1, (int) probs[2], vocab.size(), probs[0], probs[1], label);

            if (label.equals("spam") && isSpam == 1 || label.equals("ham") && isSpam == 0) {
                s += " right";
                correctLabeled++;
            } else s += " wrong";
            System.out.println(s);
        }
        return new int[]{correctLabeled, testEmails.size()};
    }


    // Method to parse email data from text file
    public static ArrayList<String> parseData(String filename) {
        ArrayList<String> emails = new ArrayList<>();
        StringBuilder emailContent = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {

            String line;
            boolean inEmail = false;

            while ((line = reader.readLine()) != null) {
                if (line.startsWith("<SUBJECT>")) {
                    inEmail = true;
                    emailContent = new StringBuilder();
                } else if (line.startsWith("</BODY>")) {
                    inEmail = false;
                    String emailText = emailContent.toString().trim().toLowerCase();
                    if (!emailText.isEmpty()) {
                        emailText = emailText.replace("</subject>", "").replace("<body>", "").trim();
                        emails.add(emailText);
                    }
                } else if (inEmail) {
                    // Split the line into words and append to email content
                    String[] words = line.split("[\\s\\n]+");
                    emailContent.append(String.join(" ", words)).append("\n");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return emails;
    }

    // Method to parse email data, ONLY the body content, from text file
    public static ArrayList<String> parseOnlyBodyData(String filename) {
        ArrayList<String> bodyEmails = new ArrayList<>();
        StringBuilder bodyEmailContent = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean inBody = false;

            while ((line = reader.readLine()) != null) {
                if (line.startsWith("<BODY>")) {
                    inBody = true;
                    bodyEmailContent = new StringBuilder();
                } else if (line.startsWith("</BODY>")) {
                    inBody = false;
                    String bodyEmailText = bodyEmailContent.toString().trim().toLowerCase();
                    bodyEmails.add(bodyEmailText);
                } else if (inBody) {
                    // Split the line into words and append to email content
                    String[] bodyWords = line.split("[\\s\\n]+");
                    bodyEmailContent.append(String.join(" ", bodyWords)).append("\n");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bodyEmails;
    }

    // Main method to execute Naive Bayes classifier
    public static void main(String[] args) {
        NaiveBayesClassifier classifier = new NaiveBayesClassifier();

        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter spam training file name: ");
        String spamTrainingFile = scanner.nextLine();

        System.out.println("Enter ham training file name: ");
        String hamTrainingFile = scanner.nextLine();

        System.out.println("Enter spam testing file name: ");
        String spamTestingFile = scanner.nextLine();

        System.out.println("Enter ham testing file name: ");
        String hamTestingFile = scanner.nextLine();

        // Train the classifier
        classifier.train(spamTrainingFile, hamTrainingFile);

        // Test the classifier
        int[] data1 = classifier.test(spamTestingFile, 1);
        int[] data2 = classifier.test(hamTestingFile, 0);
        System.out.println("Total: " + (data1[0] + data2[0]) + "/" + (data1[1] + data2[1]) + " emails classified correctly.");

    }
}
