
CREATE OR REPLACE FUNCTION validate_cnpj(cnpj_input TEXT) 
RETURNS BOOLEAN AS $$
DECLARE
    cnpj_clean TEXT;
    first_digit INTEGER;
    second_digit INTEGER;
    sum INTEGER;
    weight INTEGER;
    digit INTEGER;
    i INTEGER;
BEGIN
    
    cnpj_clean := regexp_replace(cnpj_input, '[^0-9]', '', 'g');
    
    
    IF length(cnpj_clean) != 14 THEN
        RAISE NOTICE 'CNPJ com comprimento inválido: % dígitos (deve ter 14)', length(cnpj_clean);
        RETURN FALSE;
    END IF;
    
    -- Verifica se são todos dígitos iguais (inválido)
    IF cnpj_clean ~ '^(\d)\1+$' THEN
        RAISE NOTICE 'CNPJ com todos dígitos iguais: %', cnpj_input;
        RETURN FALSE;
    END IF;
    
    
    sum := 0;
    weight := 5; -- Peso inicial para o primeiro dígito
    
    
    FOR i IN 1..12 LOOP
        sum := sum + (CAST(substring(cnpj_clean FROM i FOR 1) AS INTEGER) * weight);
        weight := weight - 1;
        IF weight < 2 THEN
            weight := 9;
        END IF;
    END LOOP;
    
    digit := 11 - (sum % 11);
    IF digit >= 10 THEN
        digit := 0;
    END IF;
    
    -- Verifica primeiro dígito verificador (13ª posição)
    IF digit != CAST(substring(cnpj_clean FROM 13 FOR 1) AS INTEGER) THEN
        RAISE NOTICE 'Primeiro dígito verificador inválido. Esperado: %, Encontrado: %', 
                     digit, CAST(substring(cnpj_clean FROM 13 FOR 1) AS INTEGER);
        RETURN FALSE;
    END IF;
    
    
    sum := 0;
    weight := 6; -- Peso inicial para o segundo dígito
    
    
    FOR i IN 1..13 LOOP
        sum := sum + (CAST(substring(cnpj_clean FROM i FOR 1) AS INTEGER) * weight);
        weight := weight - 1;
        IF weight < 2 THEN
            weight := 9;
        END IF;
    END LOOP;
    
    digit := 11 - (sum % 11);
    IF digit >= 10 THEN
        digit := 0;
    END IF;
    
    
    IF digit != CAST(substring(cnpj_clean FROM 14 FOR 1) AS INTEGER) THEN
        RAISE NOTICE 'Segundo dígito verificador inválido. Esperado: %, Encontrado: %', 
                     digit, CAST(substring(cnpj_clean FROM 14 FOR 1) AS INTEGER);
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Erro ao validar CNPJ %: %', cnpj_input, SQLERRM;
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;



CREATE OR REPLACE FUNCTION validate_ein(ein_input TEXT) 
RETURNS BOOLEAN AS $$
DECLARE
    ein_clean TEXT;
    prefix INTEGER;
    check_digit INTEGER;
    digits INTEGER[];
    i INTEGER;
    total INTEGER := 0;
    weights INTEGER[] := ARRAY[2, 3, 4, 5, 6, 7, 8, 9];
BEGIN
    ein_clean := regexp_replace(ein_input, '[^0-9]', '', 'g');
    
    IF length(ein_clean) != 9 THEN
        RETURN FALSE;
    END IF;
    
    prefix := CAST(substring(ein_clean FROM 1 FOR 2) AS INTEGER);
    check_digit := CAST(substring(ein_clean FROM 9 FOR 1) AS INTEGER);
    
    IF ein_clean ~ '^(\d)\1+$' THEN
        RETURN FALSE;
    END IF;
    
    
    IF prefix IN (00, 07, 08, 09, 17, 18, 19, 28, 29, 68, 69, 96, 97, 98, 99) THEN
        RETURN FALSE;
    END IF;
    
    
    IF NOT (
        (prefix BETWEEN 1 AND 6) OR
        (prefix BETWEEN 10 AND 16) OR
        (prefix BETWEEN 20 AND 27) OR
        (prefix BETWEEN 30 AND 39) OR
        (prefix BETWEEN 40 AND 49) OR
        (prefix BETWEEN 50 AND 59) OR
        (prefix BETWEEN 60 AND 67) OR
        (prefix BETWEEN 70 AND 90) OR
        (prefix BETWEEN 91 AND 95)
    ) THEN
        RETURN FALSE;
    END IF;
    
    
    digits := ARRAY[
        CAST(substring(ein_clean FROM 1 FOR 1) AS INTEGER),
        CAST(substring(ein_clean FROM 2 FOR 1) AS INTEGER),
        CAST(substring(ein_clean FROM 3 FOR 1) AS INTEGER),
        CAST(substring(ein_clean FROM 4 FOR 1) AS INTEGER),
        CAST(substring(ein_clean FROM 5 FOR 1) AS INTEGER),
        CAST(substring(ein_clean FROM 6 FOR 1) AS INTEGER),
        CAST(substring(ein_clean FROM 7 FOR 1) AS INTEGER),
        CAST(substring(ein_clean FROM 8 FOR 1) AS INTEGER)
    ];
    
    FOR i IN 1..8 LOOP
        total := total + (digits[i] * weights[i]);
    END LOOP;
    
    DECLARE
        remainder INTEGER;
        calculated_check INTEGER;
    BEGIN
        remainder := total % 11;
        
        IF remainder = 0 OR remainder = 1 THEN
            calculated_check := 0;
        ELSE
            calculated_check := 11 - remainder;
        END IF;
        
        IF calculated_check != check_digit THEN
            RETURN FALSE;
        END IF;
    END;
    
    RETURN TRUE;
    
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;


CREATE OR REPLACE FUNCTION format_cnpj(cnpj_input TEXT)
RETURNS TEXT AS $$
DECLARE
    cnpj_clean TEXT;
BEGIN
    IF cnpj_input IS NULL THEN
        RETURN NULL;
    END IF;
    
    cnpj_clean := regexp_replace(cnpj_input, '[^0-9]', '', 'g');
    
    IF length(cnpj_clean) != 14 THEN
        RETURN cnpj_input;
    END IF;
    
    RETURN substring(cnpj_clean from 1 for 2) || '.' ||
           substring(cnpj_clean from 3 for 3) || '.' ||
           substring(cnpj_clean from 6 for 3) || '/' ||
           substring(cnpj_clean from 9 for 4) || '-' ||
           substring(cnpj_clean from 13 for 2);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


CREATE OR REPLACE FUNCTION format_ein(ein_input TEXT)
RETURNS TEXT AS $$
DECLARE
    ein_clean TEXT;
BEGIN
    IF ein_input IS NULL OR ein_input = '' THEN
        RETURN NULL;
    END IF;
    
    ein_clean := regexp_replace(ein_input, '[^0-9]', '', 'g');
    
    IF length(ein_clean) != 9 THEN
        RETURN ein_input;
    END IF;
    
    RETURN substring(ein_clean from 1 for 2) || '-' ||
           substring(ein_clean from 3 for 7);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


CREATE OR REPLACE FUNCTION validate_organization_tax_ids()
RETURNS TRIGGER AS $$
BEGIN
    
    IF NEW.cnpj IS NOT NULL AND NEW.cnpj != '' THEN
        IF NOT validate_cnpj(NEW.cnpj) THEN
            RAISE EXCEPTION 'Invalid CNPJ: %', NEW.cnpj
            USING HINT = 'CNPJ must be 14 valid digits with correct verification digits',
                  ERRCODE = '23514';
        END IF;
        
        
        NEW.cnpj := format_cnpj(NEW.cnpj);
    END IF;
    
    
    IF NEW.ein IS NOT NULL AND NEW.ein != '' THEN
        IF NOT validate_ein(NEW.ein) THEN
            RAISE EXCEPTION 'Invalid EIN: %', NEW.ein
            USING HINT = 'EIN must be 9 valid digits with correct IRS prefix and check digit',
                  ERRCODE = '23514';
        END IF;
        
        
        NEW.ein := format_ein(NEW.ein);
    END IF;
    
    
    IF (NEW.cnpj IS NULL OR NEW.cnpj = '') AND 
       (NEW.ein IS NULL OR NEW.ein = '') THEN
        RAISE EXCEPTION 'Organization must have either CNPJ (Brazil) or EIN (US)'
        USING HINT = 'Provide at least one tax identifier',
              ERRCODE = '23514';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS trigger_validate_organization ON public.organizations;

CREATE TRIGGER trigger_validate_organization
    BEFORE INSERT OR UPDATE 
    ON public.organizations
    FOR EACH ROW
    EXECUTE FUNCTION validate_organization_tax_ids();


DO $$
BEGIN
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'public' 
                   AND table_name = 'organizations' 
                   AND column_name = 'cnpj') THEN
        ALTER TABLE public.organizations ADD COLUMN cnpj VARCHAR(18);
        RAISE NOTICE 'Coluna CNPJ adicionada à tabela organizations';
    END IF;
    
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'public' 
                   AND table_name = 'organizations' 
                   AND column_name = 'ein') THEN
        ALTER TABLE public.organizations ADD COLUMN ein VARCHAR(18);
        RAISE NOTICE 'Coluna EIN adicionada à tabela organizations';
    END IF;
    
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_schema = 'public' 
                   AND table_name = 'organizations' 
                   AND column_name = 'address') THEN
        ALTER TABLE public.organizations ADD COLUMN address VARCHAR(255);
        RAISE NOTICE 'Coluna address adicionada à tabela organizations';
    END IF;
END $$;


DO $$
BEGIN
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_schema = 'public'
                   AND table_name = 'organizations'
                   AND constraint_name = 'uni_organizations_cnpj') THEN
        ALTER TABLE public.organizations ADD CONSTRAINT uni_organizations_cnpj UNIQUE (cnpj);
        RAISE NOTICE 'Constraint UNIQUE para CNPJ adicionada';
    END IF;
    
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_schema = 'public'
                   AND table_name = 'organizations'
                   AND constraint_name = 'uni_organizations_ein') THEN
        ALTER TABLE public.organizations ADD CONSTRAINT uni_organizations_ein UNIQUE (ein);
        RAISE NOTICE 'Constraint UNIQUE para EIN adicionada';
    END IF;
END $$;


DO $$
DECLARE
    meu_cnpj TEXT := '27960568000100';
    resultado BOOLEAN;
BEGIN
    resultado := validate_cnpj(meu_cnpj);
    
    IF resultado THEN
        RAISE NOTICE '✅ CNPJ % é VÁLIDO', meu_cnpj;
    ELSE
        RAISE NOTICE '❌ CNPJ % é INVÁLIDO', meu_cnpj;
    END IF;
END $$;