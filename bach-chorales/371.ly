\version "2.10.33"

\header {
  title = "371 - Christ lag in Todesbanden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    ais4 b8 cis8 d4 e4 
    d4 cis4 b4 b8 a8 
    g4 a4 b8 a8 g4 
    fis8 e8 fis4 e4 } r4 
  e8 fis8 g4 a4 e8 fis8   %5
  g4 a4 b4 b4 
  e4 d4 e4 fis8 e8 
  d4 cis4 b4 cis4 
  d8 c8 b8 c8 d4 a4 
  g4 fis4 e2   %10
  b'4 a4 g4 fis4 
  e2. 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    fis4. ais8 b8 ais16 b16 cis4~ 
    cis8 b4 ais8 fis4 fis4~
    fis8 e4 dis16 cis16 b8 dis8 e4 
    e8 dis16 cis16 dis4 b4 } r4 
  b4 e4 e4. b8   %5
  b8 c16 d16 e4 dis4 fis4 
  b4 a4 b8 gis8 cis4~ 
  cis8 b4 ais8 fis4 g4 
  fis4 d4 d2 
  e4. d8 cis2   %10
  dis8 e8 e8 fis8 b,8 e4 dis8 
  b2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e8  d8 
    cis4 d8 e8 fis8 g4 fis8 
    fis8 b,8 cis16 d16 e8 e8 dis8 b4 
    b8 c4 b16 a16 b4 b4 
    b4. a8 g4 } r4 
  g8 a8 b4 e,4 a4   %5
  g8 a16 b16 c4 fis,4 dis'4 
  e8 fis16 g16 a8 fis8 b,8 e8 cis8 b16 ais16 
  b8 d8 g8 fis8 d4 e4 
  a,4 b8 a8 g4. fis8 
  b4 a4 a2   %10
  b4 cis8 dis8 e8 g,8 c8 b16 a16 
  gis2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    e4 d8 cis8 b8 b'4 ais8 
    b8 g8 e8 fis8 b,4 dis4 
    e4 fis4 g8 fis8 g8 a8 
    b4 b,4 e4 } r4 
  e4. d8 c8 b8 cis8 dis8   %5
  e8 d8 c4 b4 b'8 a8 
  g8 fis16 e16 fis8 a8 gis8 cis8 ais8 fis8 
  b8 g8 e8 fis8 b,4 e4 
  d4 g,8 a8 b8 c8 d4~ 
  d8 cis8 d8 fis8 a2   %10
  a8 g4 fis8 e8 c8 a8 b8 
  e2. 
  
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