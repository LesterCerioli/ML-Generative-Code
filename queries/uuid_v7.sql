-- ============================================
-- UUID v7 CORRETO PARA POSTGRESQL
-- ============================================

CREATE OR REPLACE FUNCTION uuid7() RETURNS uuid AS $$
DECLARE
    ts_ms BIGINT;
    ts_hex TEXT;
    rand_hex TEXT;
BEGIN
    -- 1. Timestamp em milissegundos (48 bits)
    ts_ms := (EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT;
    ts_hex := LPAD(TO_HEX(ts_ms), 12, '0');
    
    -- 2. Bytes aleatórios (80 bits)
    rand_hex := ENCODE(GEN_RANDOM_BYTES(10), 'hex');
    
    -- 3. UUID v7 FORMATO CORRETO:
    -- tttttttt-tttt-7ttt-8ttt-rrrrrrrrrrrr
    -- Onde '8' é FIXO para variant RFC 4122
    -- Isso É UUID v7 VÁLIDO - o '8' é CORRETO!
    
    RETURN (
        SUBSTRING(ts_hex FROM 1 FOR 8) || '-' ||
        SUBSTRING(ts_hex FROM 9 FOR 4) || '-' ||
        '7' || SUBSTRING(rand_hex FROM 1 FOR 3) || '-' ||
        '8' || SUBSTRING(rand_hex FROM 4 FOR 3) || '-' ||
        SUBSTRING(rand_hex FROM 7 FOR 12)
    )::uuid;
END;
$$ LANGUAGE plpgsql VOLATILE;

-- ============================================
-- TESTAR SE É UUID v7 VÁLIDO
-- ============================================

-- Teste 1: Gerar UUID v7
SELECT uuid7() AS uuid_v7;

-- Teste 2: Verificar formato
SELECT 
    uuid7() AS uuid,
    uuid7()::text ~ '^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-8[0-9a-f]{3}-[0-9a-f]{12}$' AS is_uuid_v7
FROM generate_series(1, 3);

-- Teste 3: Criar tabela com UUID v7
CREATE TABLE IF NOT EXISTS public.organizations (
    id UUID PRIMARY KEY DEFAULT uuid7(),
    name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT uuid7(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    organization_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    
    CONSTRAINT fk_users_organization 
        FOREIGN KEY (organization_id) 
        REFERENCES public.organizations(id) 
        ON DELETE CASCADE,
    
    CONSTRAINT unique_email_per_organization 
        UNIQUE (email, organization_id, deleted_at)
);

-- Teste 4: Inserir dados de teste
INSERT INTO public.organizations (name) 
VALUES ('Minha Organização')
ON CONFLICT (name) DO NOTHING;

-- Teste 5: Verificar que são UUID v7
SELECT 
    id,
    created_at,
    -- Extrair timestamp do UUID
    ('x' || SUBSTRING(id::text FROM 1 FOR 8) || SUBSTRING(id::text FROM 10 FOR 4))::bit(48)::bigint AS uuid_timestamp_ms,
    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint AS created_timestamp_ms
FROM public.organizations;