\version "2.10.33"

\header {
  title = "283 - Jesu, meine Freude"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    b4 b4 a4 g4 
    fis2 e2 
    b'4 cis4 d4 b4 
    e2 dis2 
    e8 fis8 g4 fis4. fis8     %5
    e1 }
  b4 b4 c4 b4 
  a4 a4 g2 
  b4 cis4 d4 b4 
  e4 d4 cis2   %10
  b2 b4 b4 
  a4 g4 fis2 
  e1 
  
}

alto = \relative c {
  \repeat volta 2 {
    r8 g''8 b8 g8 g8 fis4 e8 
    e8 dis16 cis16 dis4 e8 dis8 e8 fis8 
    g16 fis16 g8 r8 e8 a4. g16 fis16 
    g8 a16 b16 a4 b4 r8 c8 
    b8 a8 g8 b8 c8 a8 fis4     %5
    b8 gis8 e8 a8 gis2 }
  g8 a8 a8 g8 g8 a8 a8 g8 
  g4 fis4 g8 d8 e8 fis8 
  g8 d8 g4 r8 a4 g16 fis16 
  e8 g8 fis8 b8 b4 ais4   %10
  b8 fis8 b8 a8 g2~ 
  g8 fis4 e8 e8 d16 cis16 d4 
  d8 e16 d16 c16 b16 c8 b2 
  
}

tenor = \relative c {
  \repeat volta 2 {
    r8 b'8 e8 b8 e8 dis8 e8 g,8 
    c8 b16 a16 b8 a8 g8 a8 g8 fis8 
    e8 e8 r4. fis8 d'8 a8 
    b8 a16 g16 e'16 dis16 e8 fis4 r4~ 
    r8 dis8 e4. e8 d4~     %5
    d4 c16 b16 c8 b2 }
  e8 fis8 fis8 e8 e8 d8 d8 e8 
  e4 d8 c8 c8 b8 c8 a8 
  g8 b8 e8 a,4 g16 fis16 d'4~ 
  d8 cis8 d8 e16 fis16 g8 cis,8 fis8 e8   %10
  dis4 r8 b8 e8 e16 dis16 e4~ 
  e8 dis8 e8 g,8 c8 a8 fis8 b8 
  b8 gis8 e8 a4 gis8 gis4 
  
}

baixo = \relative c {
  \repeat volta 2 {
    r8 e8 g8 e8 c'8 r4 b8 
    a8 fis8 b8 b,8 e4 r4~ 
    r8 e8 a8 g8 fis8 d8 g4~ 
    g8 fis16 e16 c'4 b8 c8 b8 a8 
    g8 fis8 e8 g8 a8 fis8 b8 a8     %5
    gis8 e8 a8 a,8 e'2 }
  e8 dis8 dis8 e8 e8 fis8 g8 e8 
  c8 cis8 d4 g,2 
  r8 g'4 fis16 e16 fis8 d8 g8 fis8 
  gis8 ais8 b8 g8 e8 eis8 fis4   %10
  b,2 r8 e8 g8 e8 
  c'8 r4 b8 a8 fis8 b8 a8 
  gis8 e8 a8 a,8 e'2 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}