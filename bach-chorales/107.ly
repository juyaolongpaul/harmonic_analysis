\version "2.10.33"

\header {
  title = "107 - Herzlich lieb hab ich dich, o Herr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 ees4 
    d4 c4 bes4 aes8 g8 
    c4 c4 bes4 ees4 
    d4 c4 bes4 aes8 g8 
    c4 c4 bes4 bes4 
    c4 d4 ees8 f8 g4     %5
    f4 ees2 d4 
    ees2. } ees4 
  ees4 f4 ees8 d8 c8 d8 
  ees4 ees4 d4 d4 
  ees4 f4 ees8 d8 c8 d8   %10
  ees8 g8 f4 d4 g4 
  g4 g4 f4 ees4 
  ees4 d4 ees4 d4 
  ees4 d4 c4 bes4 
  bes4 a4 bes4 g'4   %15
  f4 ees4 d4 g4 
  aes8 g8 f4 g4 ees4 
  ees4 d4 ees4 ees4 
  ees4 ees4 f4 ees4 
  ees2. d4   %20
  ees1 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 bes4 
    bes4 g4 g4 f8 ees8 
    ees4 aes4 g4 g4 
    g8 f8 ees8 f8 g8 f8 ees8 d8 
    ees8 f8 f4 g4 g4 
    aes8 c8 bes8 aes8 g8 f8 ees4     %5
    f2 g4 aes4 
    g2. } g4 
  aes4 aes4 g4. g8 
  f8 g8 a4 bes4 bes4 
  bes4 aes4 g4 aes4   %10
  bes4 c4 bes4 bes4 
  bes4 c4 c4 aes4 
  g4 f4 g4 aes4 
  g4 f8 bes8 g4 g4 
  g4 f4 f4 ees4   %15
  f8 g16 aes16 g8 a8 b4 c4 
  c4 b8 c8 g4 g4 
  ges8 f8 f4 g4 g4 
  aes4 aes4 aes4 g4 
  f2 f2   %20
  g1 
  
}

tenor = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    f4 ees4 d4 des4 
    c8 bes8 c8 d8 ees4 g,8 aes8 
    bes4 c8 d8 ees4 aes,8 bes8 
    c8 ees8 d8 c8 d4 ees4 
    ees8 aes8 g8 f8 ees8 bes8 des4     %5
    c2 bes2 
    bes2. } bes4 
  c4 c4 c8 d8 ees8 d8 
  c4 f4 f4 f4 
  bes,4 bes4 bes4 f'4   %10
  ees8 d8 c4 f4 ees4 
  ees4 ees4 f4 f4 
  bes,4 bes4 bes4 f'4 
  bes,8 c8 d4 ees4 d4 
  ees8 d8 c4 d4 bes8 c8   %15
  d4 ees8 c8 g'4 c,4 
  f8 ees8 d8 c8 d4 c4 
  c4 bes4 bes4 des4 
  c8 d8 ees4 bes4 bes4 
  c2 bes2   %20
  bes1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 ees4 
    bes'4 c4 g4 bes4 
    aes4 aes,4 ees'4 c4 
    g'4 aes4 ees8 d8 c8 bes8 
    aes4 aes'4 g4 ees4 
    aes4 bes4 c8 d,8 ees4     %5
    aes,4 a4 bes2 
    ees2. } ees4 
  aes8 g8 f4 c4 c'8 bes8 
  a8 g8 f4 bes4 bes8 aes8 
  g8 f8 ees8 d8 ees4 f4   %10
  g4 a4 bes4 ees,4 
  ees'8 d8 c8 bes8 aes8 bes8 c8 aes8 
  bes4 bes,4 ees4 f4 
  g8 a8 bes8 g8 ees8 f8 g8 ees8 
  c4 f4 bes,4 ees4   %15
  bes'4 c4 g4 e4 
  f8 g8 aes4 b,4 c4 
  a'4 bes4 ees,4 ees4 
  aes8 bes8 c4 d,4 ees4 
  a,2 bes2   %20
  ees,1 
  
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