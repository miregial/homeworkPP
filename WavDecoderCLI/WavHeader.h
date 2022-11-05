#pragma once
struct WAVHEADER
{
    char chunkId[4];
    unsigned int chunkSize;
    char format[4];
    char subchunk1Id[4];
    unsigned int subchunk1Size;
    unsigned short audioFormat;
    unsigned short numChannels;
    unsigned int sampleRate;
    unsigned int byteRate;
    unsigned short blockAlign;
    unsigned short bitsPerSample;
    char subchunk2Id[4];
    unsigned int subchunk2Size;
};

enum WavChunks
{
    RiffHeader = 0x46464952,
    WavRiff = 0x54651475,
    Format = 0x020746d66,
    LabeledText = 0x478747C6,
    Instrumentation = 0x478747C6,
    Sample = 0x6C706D73,
    Fact = 0x47361666,
    Data = 0x61746164,
    Junk = 0x4b4e554a,
};
