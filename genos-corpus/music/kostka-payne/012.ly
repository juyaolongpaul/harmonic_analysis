\version "2.12.0"
\header {
  title = "Sonata No. 15, II"
  composer = "Haydn"
}

\new PianoStaff <<
  \new Staff {
    \time 3/4
    \relative c''' {
      \repeat volta 2 {
        \partial 4 g4\f
        g2 f4
        \grace f16 e8 d c4 c
        f2 d8 b
        c2 g'8\p e
        f2 d8 b
        c2 c4\f
        g' g g
        c2 a8 fis
        g d b' g d' b
        c2 a8 fis
        g d b' g d b
        e c b a b c
        d e g,4 fis 
        g2 
      }
      \repeat volta 2 {
        d'4
        d f8 d b d
        c e g e c e
        d4 f8 d b d
        c e g e c c'
        b a g f e d
        c4( b) g'
        g4. a8 g f
        e d c4 g'
        g4. a8 g f 
        e d c4 c
        f2 d8 b
        c2 g'8\p e
        f2 d8 b8
        c2
      }
    }
  }

  \new Staff {
    \clef bass
    \time 3/4
    \relative c { 
      \repeat volta 2 {
        \partial 4 r4
        c e g
        c2 a4
        f d g
        c, c, r
        g' g g
        c c' r
        b a g
        a d, c'
        b2 g4
        a d, c'
        b2 g4
        c, r r
        b8 c d4 d,
        g4 g'4 
      }
      \repeat volta 2 {
        r4
        f f f
        e e e8 c
        g4 g g
        c c c
        f,2 f'4
        g g, r
        c e g
        c2 r4
        c, e g
        c2 a4
        f d g
        c, c' r
        g, g g
        c4 c,
      }
    }
  }
>>