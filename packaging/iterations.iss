; Iterations - Windows installer for the meteor-cfd (ofgpu-*) solver binaries.
;
; Build with:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\iterations.iss
;
; The binaries must already be built:
;   cargo build --release            (CUDA 13, VS 2022, NVIDIA GPU)
;
; CUDA is linked statically - the only runtime dependency outside Windows itself
; is VCRUNTIME140.dll from the Visual C++ 2015-2022 redistributable, which the
; installer checks for and warns about rather than bundling.

#define AppName        "Iterations"
#define AppVersion     "0.1.0"
#define AppPublisher   "주식회사 이터레이션즈 (Iterations Co., Ltd.)"
#define AppCollaborator "주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)"
#define AppURL         "https://github.com/using76/Iteration-CFD"
#define SrcRoot        ".."

[Setup]
AppId={{7C6E1F1A-2E4B-4F55-9A2C-0E5D3B7A9C41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\Iterations
DefaultGroupName=Iterations
LicenseFile={#SrcRoot}\LICENSE
OutputDir={#SrcRoot}\dist
OutputBaseFilename=iterations-{#AppVersion}-win64-setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoCopyright=Copyright (c) 2026 {#AppPublisher} · in collaboration with {#AppCollaborator}
VersionInfoDescription=GPU-resident finite volume CFD solvers

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "{cm:AddToPath}"; GroupDescription: "{cm:PathGroup}"

[CustomMessages]
korean.AddToPath=명령 프롬프트에서 ofgpu-* 를 바로 쓸 수 있도록 PATH에 추가
korean.PathGroup=환경 변수
korean.OpenPrompt=Iterations 명령 프롬프트
korean.NoGpu=NVIDIA GPU 드라이버를 찾지 못했습니다.%n%nIterations 솔버는 CUDA 13 지원 NVIDIA GPU가 필요합니다. 드라이버가 없는 기기에서는 설치는 되지만 솔버는 실행되지 않습니다.%n%n계속하시겠습니까?
korean.NoVcRuntime=Visual C++ 2015-2022 재배포 패키지(VCRUNTIME140.dll)를 찾지 못했습니다.%n%n솔버 실행 전에 aka.ms/vs/17/release/vc_redist.x64.exe 에서 설치해 주세요.%n%n계속하시겠습니까?
english.AddToPath=Add the solvers to PATH so ofgpu-* runs from any prompt
english.PathGroup=Environment
english.OpenPrompt=Iterations command prompt
english.NoGpu=No NVIDIA GPU driver was found.%n%nThe Iterations solvers need an NVIDIA GPU with CUDA 13 support. Setup will continue, but the solvers will not run on this machine.%n%nContinue anyway?
english.NoVcRuntime=The Visual C++ 2015-2022 redistributable (VCRUNTIME140.dll) was not found.%n%nInstall it from aka.ms/vs/17/release/vc_redist.x64.exe before running the solvers.%n%nContinue anyway?

[Files]
; --- solver and tool binaries -------------------------------------------------
Source: "{#SrcRoot}\rust\target\release\ofgpu-*.exe"; DestDir: "{app}\bin"; Flags: ignoreversion

; --- licence and provenance ---------------------------------------------------
Source: "{#SrcRoot}\LICENSE";       DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\LICENSING.md";  DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\NOTICE";        DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\README.md";     DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\README.en.md";  DestDir: "{app}"; Flags: ignoreversion

; --- the case definitions the drivers read ------------------------------------
Source: "{#SrcRoot}\cases\*.jsonc"; DestDir: "{app}\cases"; Flags: ignoreversion
Source: "{#SrcRoot}\cases\README.md"; DestDir: "{app}\cases"; Flags: ignoreversion skipifsourcedoesntexist

; --- the race-car sample: a geometry and a recipe, not a mesh -----------------
; The mesh and the results are a gigabyte and are built by racecar.cmd on the
; user's own GPU; what ships is the 52 kB of STL they are built from, plus the
; two uniform fields the momentum solver needs and the mesher does not write.
Source: "{#SrcRoot}\cases\racecar.stl"; DestDir: "{app}\cases"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcRoot}\cases\racecar.md";  DestDir: "{app}\cases"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcRoot}\cases\racecar.cmd"; DestDir: "{app}\cases"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcRoot}\cases\racecar.fields\*"; DestDir: "{app}\cases\racecar.fields"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcRoot}\tools\racecar_stl.py"; DestDir: "{app}\tools"; Flags: ignoreversion skipifsourcedoesntexist

; --- the spec every discretisation is written against -------------------------
Source: "{#SrcRoot}\rust\SPEC-LIT.md";   DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcRoot}\rust\PROVENANCE.md"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcRoot}\docs\schema\*";      DestDir: "{app}\docs\schema"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\{cm:OpenPrompt}"; Filename: "{cmd}"; Parameters: "/K ""set PATH={app}\bin;%PATH% && echo Iterations {#AppVersion} && ofgpu-probe.exe"""; WorkingDir: "{app}"
Name: "{group}\README"; Filename: "{app}\README.md"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"

[Registry]
Root: HKA; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}\bin"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}\bin'))

[Run]
Filename: "{app}\bin\ofgpu-probe.exe"; Description: "{cm:LaunchProgram,ofgpu-probe}"; \
  Flags: postinstall nowait skipifsilent runasoriginaluser

[Code]
{ Append to PATH only when it is not already there, so repeated installs do not
  grow the variable without bound. }
function NeedsAddPath(Dir: string): Boolean;
var
  Existing: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Existing) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Existing) + ';') = 0;
end;

function InitializeSetup(): Boolean;
var
  Dummy: string;
begin
  Result := True;

  { nvidia-smi lives beside the driver; its absence means no usable GPU. }
  if not FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) then
    if MsgBox(CustomMessage('NoGpu'), mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
      exit;
    end;

  { CUDA is linked statically; VCRUNTIME140 is the one thing still dynamic. }
  Dummy := ExpandConstant('{sys}\VCRUNTIME140.dll');
  if not FileExists(Dummy) then
    if MsgBox(CustomMessage('NoVcRuntime'), mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
end;
