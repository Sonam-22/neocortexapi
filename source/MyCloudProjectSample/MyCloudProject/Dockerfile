#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/runtime:7.0-bullseye-slim-amd64 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0-bullseye-slim-amd64 AS build
WORKDIR /src
COPY ["temp/license.txt", "license.txt"]
WORKDIR /src/main
COPY ["nuget", "nuget"]
COPY ["nuget.config", "."]
COPY ["MyCloudProjectSample/MyCloudProject/MyCloudProject.csproj", "MyCloudProjectSample/MyCloudProject/"]
COPY ["MyCloudProjectSample/MyCloudProject.Common/MyCloudProject.Common.csproj", "MyCloudProjectSample/MyCloudProject.Common/"]
COPY ["MyCloudProjectSample/MyExperiment/MyExperiment.csproj", "MyCloudProjectSample/MyExperiment/"]
COPY ["NeoCortexApi/NeoCortexApi.csproj", "NeoCortexApi/"]
COPY ["NeoCortexArrayLib/NeoCortexArrayLib.csproj", "NeoCortexArrayLib/"]
COPY ["NeoCortexEntities/NeoCortexEntities.csproj", "NeoCortexEntities/"]
COPY ["NeoCortexUtils/NeoCortexUtils.csproj", "NeoCortexUtils/"]
COPY ["MySEProject/SDRClassifier/SDRClassifier/SDRClassifier.csproj", "MySEProject/SDRClassifier/SDRClassifier/"]
RUN dotnet restore "MyCloudProjectSample/MyCloudProject/MyCloudProject.csproj"
COPY . .
WORKDIR "/src/main/MyCloudProjectSample/MyCloudProject"
RUN dotnet build "MyCloudProject.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "MyCloudProject.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MyCloudProject.dll"]
