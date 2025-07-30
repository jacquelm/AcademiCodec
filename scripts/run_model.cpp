#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <sndfile.hh>

int main()
{
    const std::string input_path = "data/wav/input.wav";
    const std::string output_path = "data/wav/output.wav";

    // === Charger WAV avec libsndfile ===
    SndfileHandle input_file(input_path);
    if (input_file.error() || input_file.channels() != 1)
    {
        std::cerr << "Failed to read input.wav or file not mono\n";
        return 1;
    }

    int sample_rate = input_file.samplerate();
    int num_samples = input_file.frames();
    std::vector<float> audio_in(num_samples);
    input_file.readf(audio_in.data(), num_samples);

    // === Initialiser ONNX Runtime ===
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "audio_model");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "checkpoints/speech_tokenizer_static.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // === Créer le tenseur d’entrée ===
    std::vector<int64_t> input_shape = {1, 1, static_cast<int64_t>(audio_in.size())};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, audio_in.data(), audio_in.size(), input_shape.data(), input_shape.size());

    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float *output_data = output_tensors.front().GetTensorMutableData<float>();

    // === Récupérer la forme de sortie ===
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t out_samples = output_shape[2];
    std::vector<float> audio_out(output_data, output_data + out_samples);

    // === Sauvegarder WAV de sortie ===
    SndfileHandle output_file(output_path, SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, sample_rate);
    output_file.writef(audio_out.data(), out_samples);

    std::cout << "Inference complete. Saved: " << output_path << std::endl;
    return 0;
}
