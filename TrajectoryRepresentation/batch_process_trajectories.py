import os
import sys
from trajectoryProcessor import main as process_trajectories

def process_batch_trajectories():
    """
    Processa em batch os arquivos de tracking de 1_tracking.json até 100_tracking.json
    """
    
    # Configurações de diretórios
    INPUT_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\extraction\extraction2"
    OUTPUT_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\processedTrajectories"
    
    # Criar diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 Iniciando processamento em batch")
    print(f"📂 Diretório de entrada: {INPUT_DIR}")
    print(f"📁 Diretório de saída: {OUTPUT_DIR}")
    print(f"🔢 Processando arquivos de 1 a 100...")
    print("-" * 60)
    
    # Contadores para estatísticas
    processed_count = 0
    failed_count = 0
    missing_count = 0
    failed_files = []
    
    # Processar cada arquivo de 1 a 100
    for i in range(67, 101):
        # Construir caminhos
        input_filename = f"{i}.json"
        input_path = os.path.join(INPUT_DIR, input_filename)
        output_filename = f"{i}_trajectories_processed"  # .npy será adicionado automaticamente
        
        print(f"\n📋 Processando arquivo {i}/100: {input_filename}")
        
        # Verificar se arquivo de entrada existe
        if not os.path.exists(input_path):
            print(f"⚠️  Arquivo não encontrado: {input_filename}")
            missing_count += 1
            continue
        
        try:
            # Processar o arquivo usando a função main do trajectoryProcessor
            process_trajectories(input_path, OUTPUT_DIR, output_filename)
            processed_count += 1
            print(f"✅ Sucesso: {output_filename}.npy")
            
        except Exception as e:
            print(f"❌ Erro ao processar {input_filename}: {str(e)}")
            failed_count += 1
            failed_files.append(input_filename)
    
    # Relatório final
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO FINAL DO PROCESSAMENTO EM BATCH")
    print("=" * 60)
    print(f"✅ Arquivos processados com sucesso: {processed_count}")
    print(f"⚠️  Arquivos não encontrados: {missing_count}")
    print(f"❌ Arquivos com erro: {failed_count}")
    print(f"📁 Total de arquivos .npy gerados: {processed_count}")
    
    if failed_files:
        print(f"\n🚨 Arquivos que falharam:")
        for failed_file in failed_files:
            print(f"   - {failed_file}")
    
    print(f"\n📂 Todos os arquivos .npy foram salvos em:")
    print(f"   {OUTPUT_DIR}")
    
    # Verificar se diretório de saída tem os arquivos
    if os.path.exists(OUTPUT_DIR):
        npy_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
        print(f"\n📋 Arquivos .npy no diretório de saída: {len(npy_files)}")
        
        # Mostrar alguns exemplos
        if npy_files:
            print("   Exemplos:")
            for npy_file in sorted(npy_files)[:5]:
                print(f"   - {npy_file}")
            if len(npy_files) > 5:
                print(f"   ... e mais {len(npy_files) - 5} arquivos")


def check_directories():
    """
    Verifica se os diretórios necessários existem
    """
    INPUT_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\extraction\extraction2"
    OUTPUT_DIR = r"D:\UTFPR\TCC\AI-City Challenge\output_directory\processedTrajectories"
    
    print("🔍 Verificando diretórios...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Diretório de entrada não encontrado: {INPUT_DIR}")
        return False
    
    # Verificar quantos arquivos de tracking existem
    tracking_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_tracking.json') or f.endswith('.json')]
    print(f"📋 Encontrados {len(tracking_files)} arquivos *_tracking.json")
    
    # Mostrar alguns exemplos
    if tracking_files:
        print("   Exemplos:")
        for tracking_file in sorted(tracking_files)[:5]:
            print(f"   - {tracking_file}")
        if len(tracking_files) > 5:
            print(f"   ... e mais {len(tracking_files) - 5} arquivos")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"📁 Criando diretório de saída: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    else:
        print(f"✅ Diretório de saída existe: {OUTPUT_DIR}")
    
    return True


def main():
    """
    Função principal do processamento em batch
    """
    print("🎯 PROCESSADOR EM BATCH DE TRAJETÓRIAS")
    print("=" * 60)
    
    # Verificar diretórios
    if not check_directories():
        print("❌ Erro na verificação de diretórios. Abortando...")
        return
    
    # Confirmar processamento
    response = input("\n🤔 Deseja continuar com o processamento em batch? (y/n): ")
    if response.lower() not in ['y', 'yes', 'sim', 's']:
        print("⏹️  Processamento cancelado pelo usuário.")
        return
    
    # Processar todos os arquivos
    process_batch_trajectories()
    
    print("\n🎉 Processamento em batch finalizado!")


if __name__ == "__main__":
    main()