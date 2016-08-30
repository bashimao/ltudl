function convert_clsloc_synsets(matlabFile, csvFile)

  load(matlabFile);

  n = length(synsets);

  f = fopen(csvFile, "wt");

  for i = 1:n,
    fprintf(f, "%d", getfield(synsets(i), "ILSVRC2014_ID"));
    fprintf(f, ",\"%s\"", getfield(synsets(i), "WNID"));

    words = strsplit(getfield(synsets(i), "words"), ", ");
    noWords = length(words);
    fprintf(f, ",%d", noWords);
    for j = 1:noWords,
      fprintf(f, ",\"%s\"", words{j});
    end;

    gloss = strsplit(strrep(getfield(synsets(i), "gloss"), "\"", "\'"), ";");
    noGloss = length(gloss);
    fprintf(f, ",%d", noGloss);
    for j = 1:noGloss,
      fprintf(f, ",\"%s\"", gloss{j});
    end;

    noChildren = getfield(synsets(i), "num_children");
    fprintf(f, ",%d", noChildren);
    children = getfield(synsets(i), "children");
    for j = 1:noChildren,
      fprintf(f, ",%d", children(j));
    end;

    fprintf(f, ",%d", getfield(synsets(i), "wordnet_height"));
    fprintf(f, ",%d", getfield(synsets(i), "num_train_images"));
    fprintf(f, "\r\n");
  end;

  fclose(f);
end