data = load('./data/ResidualData.csv');

TrainingProcessed = ProcessRawData(data(1000:end,:));
TrainingDataMix = TrainingProcessed(randperm(size(TrainingProcessed,1)),:);
csvwrite('./data/TrainingData.csv', TrainingDataMix);
clear TrainingDataMix

TestingResi = load('../result/TestingResi.csv');
TestingProcessed = ProcessRawData(TestingResi);
csvwrite('./data/TestingData.csv', TestingProcessed);

function ProcessedData = ProcessRawData(RawData)
    data_dt = 0.01;

    num_joint = 7;
    num_feature = 1;
    num_sequence = 10;
    num_output = num_joint;

    Processed = zeros(size(RawData,1), num_joint*num_feature*num_sequence);
    recent_wrong_dt_idx = 0;
    DataIdx = 1;

    for k=num_sequence:size(RawData,1)
%        if (round(RawData(k,1) - RawData(k-1,1),3) ~= data_dt)
%             recent_wrong_dt_idx = k;
%        end
% 
%        if k < recent_wrong_dt_idx + num_sequence
%             continue
%        end

       for past_time_step = 1:num_sequence
           for joint = 1:num_joint
               Processed(DataIdx, (num_sequence-past_time_step)*num_joint*num_feature + (joint-1)*num_feature + 1) = RawData(k-past_time_step+1,joint);
           end
       end

       DataIdx = DataIdx + 1;
    end
    ProcessedData = Processed(1:DataIdx-1,:);
end